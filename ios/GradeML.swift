// GradeML.swift — on-device side of "Match the Look".
//
// UNTESTED: written on Windows without a Swift toolchain. Compile and check
// on the Mac (see docs/ios_integration.md for the verification steps).
//
// Pieces:
//   ColorCube      3D LUT in CIColorCube layout (RGBA float, red fastest),
//                  blend / intensity / apply via CIColorCubeWithColorSpace
//   LookMatcher    runs match_lut.mlmodelc -> (look LUT, match LUT)
//   Grader         runs grade_lut.mlmodelc (GradeNet): photo + style -> LUT
//                  that corrects and grades this photo (src/grademl/grade.py)
//   Editor         "Best versions": GradeNet in every style, judged by GradeNet,
//                  the editor's top 3 (mirrors top_versions() in src/grademl/grade.py)
//   PresetBank     loads the exported .lut16 preset variations + manifest
//   SubjectMask    person mask from Vision, edges refined with CIGuidedFilter
//   SubjectAware   person / background graded differently, CIBlendWithMask
//                  (mirrors src/grademl/compose.py)
//   Grain          film grain after the LUT, free on any grade (mirrors src/grademl/grain.py)
//   Surprise       photo profile + direction scoring + distinct-family pick
//                  (mirrors src/grademl/surprise.py)

import Accelerate
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreML
import Vision

// MARK: - ColorCube

struct ColorCube {
    let size: Int
    /// size^3 * 4 floats, RGBA, red index fastest, then green, then blue.
    var data: [Float]

    static func identity(size: Int) -> ColorCube {
        var d = [Float](repeating: 1, count: size * size * size * 4)
        let s = Float(size - 1)
        for b in 0..<size { for g in 0..<size { for r in 0..<size {
            let i = ((b * size + g) * size + r) * 4
            d[i] = Float(r) / s; d[i + 1] = Float(g) / s; d[i + 2] = Float(b) / s
        } } }
        return ColorCube(size: size, data: d)
    }

    /// self + t * (other - self), e.g. look -> match with the "Match" slider.
    func blended(toward other: ColorCube, amount t: Float) -> ColorCube {
        precondition(other.size == size)
        let diff = vDSP.subtract(other.data, data)      // other - self
        return ColorCube(size: size, data: vDSP.add(vDSP.multiply(t, diff), data))
    }

    /// identity + s * (self - identity): the "Intensity" slider (s may exceed 1).
    func intensity(_ s: Float) -> ColorCube {
        ColorCube.identity(size: size).blended(toward: self, amount: s).clamped()
    }

    func clamped() -> ColorCube {
        ColorCube(size: size, data: vDSP.clip(data, to: 0...1))
    }

    /// base + t * (self - base): intensity relative to another LUT. For
    /// GradeNet, base is the photo's correction (style "none"), so the slider
    /// scales the grade and keeps the correction.
    func scaled(_ t: Float, from base: ColorCube) -> ColorCube {
        base.blended(toward: self, amount: t).clamped()
    }

    /// Trilinear lookup, same as grademl.lut.apply_lut (align_corners, border).
    func lookup(_ r: Float, _ g: Float, _ b: Float) -> SIMD3<Float> {
        let s = Float(size - 1)
        let x = min(max(r, 0), 1) * s, y = min(max(g, 0), 1) * s, z = min(max(b, 0), 1) * s
        let x0 = min(Int(x), size - 2), y0 = min(Int(y), size - 2), z0 = min(Int(z), size - 2)
        let fx = x - Float(x0), fy = y - Float(y0), fz = z - Float(z0)
        func at(_ xi: Int, _ yi: Int, _ zi: Int) -> SIMD3<Float> {
            let i = ((zi * size + yi) * size + xi) * 4
            return SIMD3(data[i], data[i + 1], data[i + 2])
        }
        let c00 = at(x0, y0, z0) * (1 - fx) + at(x0 + 1, y0, z0) * fx
        let c10 = at(x0, y0 + 1, z0) * (1 - fx) + at(x0 + 1, y0 + 1, z0) * fx
        let c01 = at(x0, y0, z0 + 1) * (1 - fx) + at(x0 + 1, y0, z0 + 1) * fx
        let c11 = at(x0, y0 + 1, z0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1, z0 + 1) * fx
        return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz
    }

    /// Apply to a small RGB float image (e.g. a Thumb) on the CPU.
    func apply(to t: Thumb) -> Thumb {
        var out = t
        for i in 0..<(t.width * t.height) {
            let v = lookup(t.rgb[i * 3], t.rgb[i * 3 + 1], t.rgb[i * 3 + 2])
            out.rgb[i * 3] = v.x; out.rgb[i * 3 + 1] = v.y; out.rgb[i * 3 + 2] = v.z
        }
        return out
    }

    /// (1, 3, D, D, D) Core ML output with layout lut[c, b, g, r] -> RGBA, red fastest.
    /// Multi-index subscripting respects the array's strides (~5 ms for 33^3).
    static func from(_ a: MLMultiArray) -> ColorCube {
        let d = a.shape[4].intValue
        var data = [Float](repeating: 1, count: d * d * d * 4)
        for c in 0..<3 { for b in 0..<d { for g in 0..<d { for r in 0..<d {
            let v = a[[0, c, b, g, r] as [NSNumber]].floatValue
            data[((b * d + g) * d + r) * 4 + c] = v
        } } } }
        return ColorCube(size: d, data: data).clamped()
    }
}

// MARK: - Thumb

/// A 256x256 sRGB float copy of a photo (squashed, like the models' inputs),
/// for scoring candidates on the CPU. rgb: interleaved R, G, B in 0..1, top row first.
struct Thumb {
    let width: Int, height: Int
    var rgb: [Float]

    init(_ image: CGImage, width: Int = 256, height: Int = 256) {
        self.width = width; self.height = height
        var px = [UInt8](repeating: 0, count: width * height * 4)
        let ctx = CGContext(data: &px, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        rgb = (0..<(width * height * 3)).map { Float(px[($0 / 3) * 4 + $0 % 3]) / 255 }
    }

    /// Back to a CGImage, to feed a Core ML model.
    func cgImage() -> CGImage {
        var px = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) { for c in 0..<3 {
            px[i * 4 + c] = UInt8((min(max(rgb[i * 3 + c], 0), 1) * 255).rounded())
        } }
        let ctx = CGContext(data: &px, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        return ctx.makeImage()!
    }

    /// Mean absolute difference in % (same as (a - b).abs().mean() * 100).
    func meanDifference(_ o: Thumb) -> Double {
        Double(vDSP.meanMagnitude(vDSP.subtract(rgb, o.rgb))) * 100
    }

    /// Share of pixels blown to white or crushed to black (grademl.grade.clipped).
    var clipped: Double {
        var n = 0
        for i in 0..<(width * height) {
            let r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2]
            if min(r, g, b) > 0.97 || max(r, g, b) < 0.02 { n += 1 }
        }
        return Double(n) / Double(width * height)
    }
}

    /// Apply to a full-resolution image. The LUT maps sRGB-encoded values,
    /// so the cube's color space must be (non-linear) sRGB.
    func apply(to image: CIImage) -> CIImage {
        let f = CIFilter.colorCubeWithColorSpace()
        f.inputImage = image
        f.cubeDimension = Float(size)
        f.cubeData = data.withUnsafeBufferPointer { Data(buffer: $0) }
        f.colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        return f.outputImage ?? image
    }
}

// MARK: - Core ML

final class LookMatcher {
    private let model: MLModel

    init(computeUnits: MLComputeUnits = .all) throws {
        guard let url = Bundle.main.url(forResource: "match_lut", withExtension: "mlmodelc") else {
            throw NSError(domain: "GradeML", code: 1, userInfo: [NSLocalizedDescriptionKey: "match_lut.mlmodelc missing"])
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: config)
    }

    /// Both images are squashed to 256x256 (as in training: the model only
    /// needs color statistics, aspect ratio does not matter).
    func predict(photo: CGImage, reference: CGImage) throws -> (look: ColorCube, match: ColorCube) {
        let opts: [MLFeatureValue.ImageOption: Any] = [
            .cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue
        ]
        func feature(_ img: CGImage) throws -> MLFeatureValue {
            try MLFeatureValue(cgImage: img, pixelsWide: 256, pixelsHigh: 256,
                               pixelFormatType: kCVPixelFormatType_32BGRA, options: opts)
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "content": try feature(photo), "reference": try feature(reference),
        ])
        let out = try model.prediction(from: input)
        guard let look = out.featureValue(for: "lut_look")?.multiArrayValue,
              let match = out.featureValue(for: "lut_match")?.multiArrayValue else {
            throw NSError(domain: "GradeML", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing outputs"])
        }
        return (ColorCube.from(look), ColorCube.from(match))
    }
}

// MARK: - GradeNet

/// GradeNet: photo + style -> LUT that corrects and grades this photo.
/// Bundle grade_lut.mlmodelc and grade_lut.styles.json (scripts/export_coreml.py --model grade).
final class Grader {
    /// "none" (correction only) first, then the directions; the one-hot order of the model.
    let styles: [String]
    private let model: MLModel

    init(computeUnits: MLComputeUnits = .all) throws {
        guard let url = Bundle.main.url(forResource: "grade_lut", withExtension: "mlmodelc"),
              let stylesURL = Bundle.main.url(forResource: "grade_lut.styles", withExtension: "json") else {
            throw NSError(domain: "GradeML", code: 4, userInfo: [NSLocalizedDescriptionKey: "grade_lut model or styles missing"])
        }
        styles = try JSONDecoder().decode([String].self, from: Data(contentsOf: stylesURL))
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: config)
    }

    /// One LUT per (photo, style) pair, batched.
    func predict(_ photos: [CGImage], styles names: [String]) throws -> [ColorCube] {
        precondition(photos.count == names.count)
        let opts: [MLFeatureValue.ImageOption: Any] = [.cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue]
        let inputs: [MLFeatureProvider] = try zip(photos, names).map { photo, name in
            guard let k = styles.firstIndex(of: name) else {
                throw NSError(domain: "GradeML", code: 5, userInfo: [NSLocalizedDescriptionKey: "unknown style \(name)"])
            }
            let onehot = try MLMultiArray(shape: [1, NSNumber(value: styles.count)], dataType: .float32)
            for i in 0..<styles.count { onehot[i] = i == k ? 1 : 0 }
            return try MLDictionaryFeatureProvider(dictionary: [
                "content": try MLFeatureValue(cgImage: photo, pixelsWide: 256, pixelsHigh: 256,
                                              pixelFormatType: kCVPixelFormatType_32BGRA, options: opts),
                "style": MLFeatureValue(multiArray: onehot),
            ])
        }
        let out = try model.predictions(fromBatch: MLArrayBatchProvider(array: inputs))
        return try (0..<out.count).map { i in
            guard let lut = out.features(at: i).featureValue(for: "lut")?.multiArrayValue else {
                throw NSError(domain: "GradeML", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing lut output"])
            }
            return ColorCube.from(lut)
        }
    }

    /// The photo's correction and its LUT in each style (scale a style with
    /// `lut.scaled(t, from: correction)`).
    func grade(_ photo: CGImage, styles names: [String]) throws -> (correction: ColorCube, looks: [String: ColorCube]) {
        let luts = try predict(Array(repeating: photo, count: names.count + 1), styles: ["none"] + names)
        return (luts[0], Dictionary(uniqueKeysWithValues: zip(names, luts.dropFirst())))
    }
}

// MARK: - Presets

struct PresetBank {
    struct Direction { let name: String; let family: String; let files: [String] }
    let size: Int
    let directions: [String: Direction]
    /// Subject-aware defaults exported with the presets (manifest "subject_aware").
    let subjectAware: SubjectAware
    private let backgroundPopFile: String?
    private let folder: URL

    /// folder: a bundled "presets" directory produced by scripts/export_presets.py
    init(folder: URL) throws {
        self.folder = folder
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("manifest.json"))) as! [String: Any]
        size = json["lut_size"] as! Int
        var dirs: [String: Direction] = [:]
        for (name, v) in json["directions"] as! [String: [String: Any]] {
            dirs[name] = Direction(name: name, family: v["family"] as! String, files: v["files"] as! [String])
        }
        directions = dirs
        let sa = json["subject_aware"] as? [String: Any] ?? [:]
        backgroundPopFile = sa["background_pop_file"] as? String
        var defaults = SubjectAware()
        if let v = sa["subject_strength"] as? Double { defaults.subjectStrength = Float(v) }
        if let v = sa["background_strength"] as? Double { defaults.backgroundStrength = Float(v) }
        if let v = sa["background_pop"] as? Double { defaults.backgroundPop = Float(v) }
        if let v = sa["min_coverage"] as? Double { defaults.minCoverage = v }
        subjectAware = defaults
    }

    func cube(direction: String, variation: Int) throws -> ColorCube {
        let files = directions[direction]!.files
        return try load(files[variation % files.count])
    }

    /// The background LUT of subject-aware grading (strength 1; scale it with intensity()).
    func backgroundPop() throws -> ColorCube {
        guard let file = backgroundPopFile else {
            throw NSError(domain: "GradeML", code: 3, userInfo: [NSLocalizedDescriptionKey:
                "manifest has no subject_aware entry: re-run scripts/export_presets.py"])
        }
        return try load(file)
    }

    private func load(_ file: String) throws -> ColorCube {
        let raw = try Data(contentsOf: folder.appendingPathComponent(file))
        let n = size * size * size
        var data = [Float](repeating: 1, count: n * 4)
        raw.withUnsafeBytes { (buf: UnsafeRawBufferPointer) in
            let h = buf.bindMemory(to: Float16.self)  // little-endian float16, RGB
            for i in 0..<n { for c in 0..<3 { data[i * 4 + c] = Float(h[i * 3 + c]) } }
        }
        return ColorCube(size: size, data: data)
    }
}

// MARK: - Subject-aware grading

enum SubjectMask {
    /// Soft person mask covering `image.extent` (white = person), or nil when
    /// people cover less than `minCoverage` of the frame (the photo then gets
    /// the plain look). Pass an already oriented image (`oriented(_:)`).
    ///
    /// Run once per photo and keep the result: slider changes only re-render
    /// the Core Image graph.
    static func person(in image: CIImage, context: CIContext, minCoverage: Double = 0.01,
                       quality: VNGeneratePersonSegmentationRequest.QualityLevel = .accurate) throws -> CIImage? {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = quality
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        try VNImageRequestHandler(ciImage: image, options: [:]).perform([request])
        guard let buffer = request.results?.first?.pixelBuffer else { return nil }
        // A mask is data, not a color: no color management, values used as is.
        let coarse = CIImage(cvPixelBuffer: buffer, options: [.colorSpace: NSNull()])
        guard coverage(of: coarse, context: context) >= minCoverage else { return nil }
        return refine(coarse, to: image)
    }

    /// Same as grademl.mask.refine_mask: guided filter at `work` px (long
    /// side) with a radius of 3% of that, then scaled to the photo. Falls back
    /// to a plain upscale if CIGuidedFilter is unavailable.
    static func refine(_ coarse: CIImage, to image: CIImage, work: CGFloat = 1024,
                       radius: CGFloat = 0.03, epsilon: CGFloat = 1e-4) -> CIImage {
        let ext = image.extent
        let s = min(1, work / max(ext.width, ext.height))
        let guide = fit(image, to: CGRect(x: 0, y: 0, width: (ext.width * s).rounded(), height: (ext.height * s).rounded()))
        var mask = fit(coarse, to: guide.extent)
        if let f = CIFilter(name: "CIGuidedFilter") {
            f.setValue(mask, forKey: kCIInputImageKey)
            f.setValue(guide, forKey: "inputGuideImage")
            f.setValue(max(1, (radius * max(guide.extent.width, guide.extent.height)).rounded()), forKey: kCIInputRadiusKey)
            f.setValue(epsilon, forKey: "inputEpsilon")
            if let out = f.outputImage { mask = out.cropped(to: guide.extent) }
        }
        return fit(mask.applyingFilter("CIColorClamp"), to: ext)
    }

    /// Scale and move `image` so its extent becomes `rect`.
    private static func fit(_ image: CIImage, to rect: CGRect) -> CIImage {
        let e = image.extent
        return image
            .transformed(by: CGAffineTransform(translationX: -e.minX, y: -e.minY))
            .transformed(by: CGAffineTransform(scaleX: rect.width / e.width, y: rect.height / e.height))
            .transformed(by: CGAffineTransform(translationX: rect.minX, y: rect.minY))
            .cropped(to: rect)
    }

    /// Mean mask value = share of the frame covered by people.
    static func coverage(of mask: CIImage, context: CIContext) -> Double {
        let f = CIFilter.areaAverage()
        f.inputImage = mask
        f.extent = mask.extent
        guard let avg = f.outputImage else { return 0 }
        var px = [Float](repeating: 0, count: 4)
        context.render(avg, toBitmap: &px, rowBytes: 16, bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
                       format: .RGBAf, colorSpace: nil)
        return Double(px[0])
    }
}

/// One look, graded differently on the person and on the background:
///   subject    = look at intensity * subjectStrength
///   background = look at intensity * backgroundStrength, then backgroundPop
///   out        = blend(subject, background, mask)
/// Mirrors subject_aware_grade() in src/grademl/compose.py — keep in sync.
struct SubjectAware {
    var subjectStrength: Float = 0.75
    var backgroundStrength: Float = 1.0
    var backgroundPop: Float = 0.6
    var minCoverage: Double = 0.01

    /// mask: from SubjectMask.person (nil = no person: plain look).
    /// pop: PresetBank.backgroundPop(). Works with any look: presets,
    /// Surprise picks, LookMatcher predictions.
    /// base: what intensity 0 means (nil = identity; GradeNet: the correction).
    func render(_ image: CIImage, look: ColorCube, intensity: Float, mask: CIImage?, pop: ColorCube,
                base: ColorCube? = nil) -> CIImage {
        func at(_ t: Float) -> ColorCube { base.map { look.scaled(t, from: $0) } ?? look.intensity(t) }
        guard let mask else { return at(intensity).apply(to: image) }
        let subject = at(intensity * subjectStrength).apply(to: image)
        let background = pop.intensity(backgroundPop)
            .apply(to: at(intensity * backgroundStrength).apply(to: image))
        // Blends in Core Image's linear working space, like compose.blend().
        let f = CIFilter.blendWithMask()
        f.inputImage = subject
        f.backgroundImage = background
        f.maskImage = mask
        return f.outputImage?.cropped(to: image.extent) ?? subject
    }
}

// MARK: - Best version

/// One of the editor's choices. Render with `render(_:mask:pop:)`, then Grain if wanted.
struct Edit {
    let rank: Int                // the editor's ranking: 1 = its pick, 2, 3 = next choices
    let correction: ColorCube
    let look: ColorCube          // GradeNet LUT of the chosen style (correction included)
    let style: String            // "none" = correction only
    let intensity: Float
    let subject: SubjectAware?   // non-nil when a person is in the frame
    let reasons: [String]
    let ranked: [(style: String, intensity: Float, score: Double)]

    var cube: ColorCube { look.scaled(intensity, from: correction) }

    func render(_ image: CIImage, mask: CIImage?, pop: ColorCube) -> CIImage {
        guard let subject, let mask else { return cube.apply(to: image) }
        return subject.render(image, look: look, intensity: intensity, mask: mask, pop: pop, base: correction)
    }
}

/// Mirrors top_versions() in src/grademl/grade.py — keep the scoring in sync.
enum Editor {
    struct Weights {
        var critic = 1.0, fit = 0.5, bold = 0.35, boldCap = 8.0, clip = 0.5, skin = 1.0
    }

    /// The editor's top n choices, ranked by its score, one per style (each at
    /// its best intensity). photo: oriented CGImage; mask: SubjectMask.person
    /// (nil = no person). ~60 GradeNet predictions on 256 px thumbnails plus
    /// CPU scoring: run off the main thread.
    static func topVersions(photo: CGImage, grader: Grader, mask: CIImage?, context: CIContext, n: Int = 3,
                            intensities: [Float] = [0.8, 1.0, 1.25], weights w: Weights = Weights()) throws -> [Edit] {
        let directions = grader.styles.filter { $0 != "none" }
        let (correction, looks) = try grader.grade(photo, styles: directions)
        let thumb = Thumb(photo)
        let corrected = correction.apply(to: thumb)

        var specs: [(style: String, t: Float, cube: ColorCube)] = [("none", 1, correction)]
        for d in directions { for t in intensities { specs.append((d, t, looks[d]!.scaled(t, from: correction))) } }
        let cands = specs.map { $0.cube.apply(to: thumb) }

        // Critic: how much GradeNet ("none") would still correct each candidate.
        let fixes = try grader.predict(cands.map { $0.cgImage() }, styles: Array(repeating: "none", count: cands.count))
        let critic = zip(fixes, cands).map { fix, c in fix.apply(to: c).meanDifference(c) }

        let prior = Surprise.scores(PhotoProfile(image: corrected.cgImage()))
        let top = prior.values.max() ?? 1
        let skinPixels = mask.flatMap { skinIndices(corrected, mask: $0, context: context) } ?? []
        let baseClip = corrected.clipped

        var ranked: [(style: String, intensity: Float, score: Double, skin: Double)] = []
        for (i, spec) in specs.enumerated() {
            let fit = spec.style == "none" ? 0 : w.fit * log((prior[spec.style] ?? top) / top)
            let bold = w.bold * min(cands[i].meanDifference(corrected), w.boldCap)
            let clip = w.clip * max(0, cands[i].clipped - baseClip) * 100
            let skin = w.skin * skinPenalty(corrected, cands[i], skinPixels)
            ranked.append((spec.style, spec.t, -w.critic * critic[i] + fit + bold - clip - skin, skin))
        }
        ranked.sort { $0.score > $1.score }

        var picks: [(style: String, intensity: Float, score: Double, skin: Double)] = []
        for c in ranked where picks.count < n && !picks.contains(where: { $0.style == c.style }) { picks.append(c) }
        let summary = correctionSummary(thumb, corrected)
        return picks.enumerated().map { k, pick in
            let rank = k + 1
            let subject: SubjectAware? = mask == nil ? nil : {
                var s = SubjectAware(); s.subjectStrength = pick.skin > 0.5 * w.skin ? 0.6 : 0.75; return s
            }()
            var reasons = [summary]
            let name = pick.style.replacingOccurrences(of: "_", with: " ")
            reasons.append(pick.style == "none" ? "Grade: kept it clean"
                           : "Grade: \(name) at \(Int(pick.intensity * 100))%, edited by GradeNet for this photo")
            reasons.append(rank == 1 ? "Editor's choice #1"
                           : String(format: "Editor's choice #%d (score %.2f vs %.2f for #1)", rank, pick.score, picks[0].score))
            if pick.skin > 0.05 { reasons.append("Skin: kept close to natural") }
            if let subject { reasons.append("Person found: grade at \(Int(subject.subjectStrength * 100))% on the subject, background a bit darker and calmer") }
            return Edit(rank: rank, correction: correction, look: pick.style == "none" ? correction : looks[pick.style]!,
                        style: pick.style, intensity: pick.intensity, subject: subject, reasons: reasons,
                        ranked: ranked.map { ($0.style, $0.intensity, $0.score) })
        }
    }

    /// Pixels (thumbnail indices) that are person and skin-colored (grade._skin).
    private static func skinIndices(_ t: Thumb, mask: CIImage, context: CIContext) -> [Int] {
        let e = mask.extent
        let small = mask.transformed(by: CGAffineTransform(scaleX: CGFloat(t.width) / e.width, y: CGFloat(t.height) / e.height))
            .transformed(by: CGAffineTransform(translationX: -e.minX * CGFloat(t.width) / e.width,
                                               y: -e.minY * CGFloat(t.height) / e.height))
        var m = [Float](repeating: 0, count: t.width * t.height * 4)
        // CIContext rows are bottom-up relative to Thumb's top-down rows: flip while reading.
        context.render(small, toBitmap: &m, rowBytes: t.width * 16,
                       bounds: CGRect(x: 0, y: 0, width: t.width, height: t.height), format: .RGBAf, colorSpace: nil)
        var out: [Int] = []
        for y in 0..<t.height { for x in 0..<t.width {
            let i = y * t.width + x
            guard m[((t.height - 1 - y) * t.width + x) * 4] > 0.5 else { continue }
            let (L, a, b) = PhotoProfile.lab(Double(t.rgb[i * 3]), Double(t.rgb[i * 3 + 1]), Double(t.rgb[i * 3 + 2]))
            let C = (a * a + b * b).squareRoot()
            var h = atan2(b, a) * 180 / .pi; if h < 0 { h += 360 }
            if h > 20, h < 80, C > 8, C < 50, L > 25, L < 92 { out.append(i) }
        } }
        return out
    }

    /// grade.skin_penalty: hue drift beyond 6 degrees per 10 degrees, chroma change beyond 25% per 25%.
    private static func skinPenalty(_ ref: Thumb, _ cand: Thumb, _ idx: [Int]) -> Double {
        guard idx.count >= 20 else { return 0 }
        func meanLab(_ t: Thumb) -> (Double, Double) {
            var sa = 0.0, sb = 0.0
            for i in idx {
                let (_, a, b) = PhotoProfile.lab(Double(t.rgb[i * 3]), Double(t.rgb[i * 3 + 1]), Double(t.rgb[i * 3 + 2]))
                sa += a; sb += b
            }
            return (sa / Double(idx.count), sb / Double(idx.count))
        }
        let (ra, rb) = meanLab(ref), (qa, qb) = meanLab(cand)
        var dh = abs(atan2(qb, qa) - atan2(rb, ra)) * 180 / .pi
        dh = min(dh, 360 - dh)
        let dc = abs((qa * qa + qb * qb).squareRoot() / max((ra * ra + rb * rb).squareRoot(), 1e-3) - 1)
        return max(0, dh - 6) / 10 + max(0, dc - 0.25) / 0.25
    }

    private static func correctionSummary(_ before: Thumb, _ after: Thumb) -> String {
        func means(_ t: Thumb) -> (L: Double, b: Double, C: Double) {
            var L = 0.0, B = 0.0, C = 0.0
            let n = t.width * t.height
            for i in 0..<n {
                let (l, a, b) = PhotoProfile.lab(Double(t.rgb[i * 3]), Double(t.rgb[i * 3 + 1]), Double(t.rgb[i * 3 + 2]))
                L += l; B += b; C += (a * a + b * b).squareRoot()
            }
            return (L / Double(n), B / Double(n), C / Double(n))
        }
        let x = means(before), y = means(after)
        var fixes: [String] = []
        if abs(y.L - x.L) > 2 { fixes.append(y.L > x.L ? "brighter" : "darker") }
        if abs(y.b - x.b) > 1.5 { fixes.append(y.b > x.b ? "warmer white balance" : "cooler white balance") }
        let ds = y.C / max(x.C, 1e-3) - 1
        if abs(ds) > 0.08 { fixes.append(ds > 0 ? "more saturation" : "less saturation") }
        return "Correction: " + (fixes.isEmpty ? "the photo was already well balanced" : fixes.joined(separator: ", "))
    }
}

// MARK: - Grain

enum Grain {
    /// Film grain, applied last (after the LUT and the subject-aware blend).
    /// Same method as grademl.grain.add_grain: uniform noise on a grid of
    /// ~`size` px clumps (size * short side / 1500), linearly upscaled,
    /// centered on mid-gray with std `amount`, soft-light blended on the
    /// sRGB-encoded image. The noise pattern differs from Python's; strength,
    /// clump size and tonal response match. Free on any grade: amount and
    /// size are user controls (amount 0.03-0.08 reads as film).
    static func apply(to image: CIImage, amount: Float, size: Float) -> CIImage {
        guard amount > 0, let random = CIFilter.randomGenerator().outputImage else { return image }
        let e = image.extent
        let px = max(1, CGFloat(size) * min(e.width, e.height) / 1500)
        let gain = CGFloat(amount) / 0.288675  // std of U(0, 1)
        let bias = 0.5 - 0.5 * gain
        let gray = CIVector(x: gain, y: 0, z: 0, w: 0)  // red channel of the noise -> R, G, B
        let noise = random
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": gray, "inputGVector": gray, "inputBVector": gray,
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputBiasVector": CIVector(x: bias, y: bias, z: bias, w: 1),
            ])
            .applyingFilter("CIColorClamp")
            .transformed(by: CGAffineTransform(scaleX: px, y: px))  // linear sampling: soft clumps
            .transformed(by: CGAffineTransform(translationX: e.minX, y: e.minY))
            .cropped(to: e)
        // Blend on sRGB-encoded values, like Python (Core Image works in linear light).
        let f = CIFilter.softLightBlendMode()
        f.inputImage = noise
        f.backgroundImage = image.applyingFilter("CILinearToSRGBToneCurve")
        guard let out = f.outputImage else { return image }
        return out.applyingFilter("CISRGBToneCurveToLinear").cropped(to: e)
    }
}

// MARK: - Surprise me

struct PhotoProfile {
    var brightness = 0.0, dark = 0.0, sky = 0.0, green = 0.0, warm = 0.0, colorfulness = 0.0

    /// Same measurements as grademl.surprise.photo_profile, on a 96x96 copy.
    init(image: CGImage) {
        let w = 96, h = 96
        var px = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.interpolationQuality = .medium
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        let n = Double(w * h)
        for i in 0..<(w * h) {
            let (L, a, b) = Self.lab(Double(px[i * 4]) / 255, Double(px[i * 4 + 1]) / 255, Double(px[i * 4 + 2]) / 255)
            let C = (a * a + b * b).squareRoot()
            var hue = atan2(b, a) * 180 / .pi; if hue < 0 { hue += 360 }
            brightness += L / 100 / n
            colorfulness += C / 40 / n
            if L < 20 { dark += 1 / n }
            if C > 10, hue > 220, hue < 300, L > 45 { sky += 1 / n }
            if C > 10, hue > 100, hue < 170 { green += 1 / n }
            if C > 6, hue < 100 || hue > 340 { warm += 1 / n }
        }
    }

    static func lab(_ r: Double, _ g: Double, _ b: Double) -> (Double, Double, Double) {
        func lin(_ v: Double) -> Double { v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4) }
        let (R, G, B) = (lin(r), lin(g), lin(b))
        let x = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047
        let y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B
        let z = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / 1.08883
        func f(_ t: Double) -> Double { t > 216.0 / 24389 ? cbrt(t) : (24389.0 / 27 * t + 16) / 116 }
        return (116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z)))
    }
}

enum Surprise {
    /// Mirrors direction_scores() in src/grademl/surprise.py — keep in sync.
    static func scores(_ p: PhotoProfile) -> [String: Double] {
        let night = p.brightness < 0.3 || p.dark > 0.4
        let nightF = night ? 1.0 : 0.0
        var s: [String: Double] = [:]
        for d in ["teal_orange", "golden_hour", "portra", "fuji_chrome", "cinestill_night", "moody_green",
                  "matte_pastel", "bleach_bypass", "vibrant_travel", "autumn", "neon_night", "bw_toned",
                  "vintage_70s", "nordic_cool"] { s[d] = 1 }
        s["golden_hour"]! += 2.0 * p.sky + 1.5 * p.warm + (night ? 0 : 0.5)
        s["teal_orange"]! += 3.0 * min(p.sky + p.green, 1) + 1.5 * p.warm
        s["autumn"]! += 3.0 * p.green
        s["moody_green"]! += 2.5 * p.green + (p.brightness < 0.45 ? 0.5 : 0)
        s["cinestill_night"]! += 2.5 * nightF + 1.0 * p.warm * nightF
        s["neon_night"]! += 2.0 * nightF
        s["nordic_cool"]! += 1.5 * p.sky + (p.colorfulness < 0.5 ? 0.5 : 0)
        s["vibrant_travel"]! += 1.5 * p.sky + 1.0 * p.green - (p.colorfulness > 0.9 ? 1 : 0)
        s["portra"]! += 1.0 * p.warm + 0.5
        s["fuji_chrome"]! += 0.8
        s["vintage_70s"]! += 0.8 * p.warm
        s["matte_pastel"]! += (p.brightness > 0.55 ? 1 : 0) - nightF
        s["bw_toned"]! += p.colorfulness < 0.15 ? 1.2 : 0
        s["bleach_bypass"]! += p.colorfulness < 0.2 ? 0.8 : 0
        if !night { s["cinestill_night"]! -= 0.5; s["neon_night"]! -= 0.7 }
        return s.mapValues { max($0, 0.05) }
    }

    struct Pick { let direction: String; let variation: Int; let intensity: Float; let cube: ColorCube }

    /// n bold looks from different families. Intensity 1.0-1.3 on top of the
    /// preset (presets are exported at strength 1.0).
    static func pick(for image: CGImage, bank: PresetBank, n: Int = 3, temperature: Double = 0.4,
                     using rng: inout some RandomNumberGenerator) throws -> [Pick] {
        let scores = scores(PhotoProfile(image: image))
        var pool = scores.map { (name: $0.key, w: pow($0.value, 1 / temperature)) }
        var usedFamilies = Set<String>()
        var picks: [Pick] = []
        while picks.count < n, !pool.isEmpty {
            // Weighted draw without replacement.
            var x = Double.random(in: 0..<pool.reduce(0) { $0 + $1.w }, using: &rng)
            let idx = pool.firstIndex { x -= $0.w; return x < 0 } ?? pool.count - 1
            let name = pool.remove(at: idx).name
            guard let dir = bank.directions[name], !usedFamilies.contains(dir.family) else { continue }
            usedFamilies.insert(dir.family)
            let variation = Int.random(in: 0..<dir.files.count, using: &rng)
            let s = Float.random(in: 1.0...1.3, using: &rng)
            picks.append(Pick(direction: name, variation: variation, intensity: s,
                              cube: try bank.cube(direction: name, variation: variation).intensity(s)))
        }
        return picks
    }
}
