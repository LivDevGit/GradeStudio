// GradeML.swift — on-device side of "Match the Look".
//
// UNTESTED: written on Windows without a Swift toolchain. Compile and check
// on the Mac (see docs/ios_integration.md for the verification steps).
//
// Pieces:
//   ColorCube      3D LUT in CIColorCube layout (RGBA float, red fastest),
//                  blend / intensity / apply via CIColorCubeWithColorSpace
//   LookMatcher    runs match_lut.mlmodelc -> (look LUT, match LUT)
//   PresetBank     loads the exported .lut16 preset variations + manifest
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
        return (Self.cube(from: look), Self.cube(from: match))
    }

    /// (1, 3, D, D, D) with layout lut[c, b, g, r] -> CIColorCube RGBA, red fastest.
    /// Multi-index subscripting respects the array's strides (~5 ms for 33^3).
    private static func cube(from a: MLMultiArray) -> ColorCube {
        let d = a.shape[4].intValue
        var data = [Float](repeating: 1, count: d * d * d * 4)
        for c in 0..<3 { for b in 0..<d { for g in 0..<d { for r in 0..<d {
            let v = a[[0, c, b, g, r] as [NSNumber]].floatValue
            data[((b * d + g) * d + r) * 4 + c] = v
        } } } }
        return ColorCube(size: d, data: data).clamped()
    }
}

// MARK: - Presets

struct PresetBank {
    struct Direction { let name: String; let family: String; let files: [String] }
    let size: Int
    let directions: [String: Direction]
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
    }

    func cube(direction: String, variation: Int) throws -> ColorCube {
        let files = directions[direction]!.files
        let raw = try Data(contentsOf: folder.appendingPathComponent(files[variation % files.count]))
        let n = size * size * size
        var data = [Float](repeating: 1, count: n * 4)
        raw.withUnsafeBytes { (buf: UnsafeRawBufferPointer) in
            let h = buf.bindMemory(to: Float16.self)  // little-endian float16, RGB
            for i in 0..<n { for c in 0..<3 { data[i * 4 + c] = Float(h[i * 3 + c]) } }
        }
        return ColorCube(size: size, data: data)
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
