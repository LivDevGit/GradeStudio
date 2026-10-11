// LocalGrade.swift — GradeNet-L on device: K LUTs blended per pixel by a
// bilateral grid (mirrors src/grademl/local.py).
//
// UNTESTED: written on Windows without a Swift toolchain. Compile and check
// on the Mac (docs/ios_integration.md). Needs GradeML.swift (ColorCube) and
// LocalGrade.ci.metal in the same target.
//
// Bundle grade_local_lut.mlmodelc + grade_local_lut.styles.json
// (scripts/export_coreml.py --model grade_local).
//
//   let grader = try LocalGrader()
//   let edit = try grader.predict(photo, style: "golden_hour")
//   let out = try edit.render(CIImage(cgImage: photo))          // full resolution
//   let half = try edit.render(image, intensity: 0.5)           // slider: blends toward the input

import CoreImage
import CoreImage.CIFilterBuiltins
import CoreML
import Vision

/// K LUTs + a packed bilateral grid; render() applies them to any resolution.
struct LocalEdit {
    let luts: [ColorCube]       // K <= 4
    let gridImage: CIImage      // (G * gw) x gh RGBA float, see LocalGrader.pack
    let gridBins: Int
    let gridSize: (w: Int, h: Int)

    private static var kernel: CIKernel? = {
        guard let url = Bundle.main.url(forResource: "default", withExtension: "metallib"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? CIKernel(functionName: "localBlend", fromMetalLibraryData: data)
    }()

    /// intensity 1 = the model's edit, 0 = the input photo (blend on top of the result).
    func render(_ image: CIImage, intensity: Float = 1) throws -> CIImage {
        guard let kernel = LocalEdit.kernel else {
            throw NSError(domain: "GradeML", code: 10, userInfo: [NSLocalizedDescriptionKey: "localBlend kernel missing"])
        }
        let e = image.extent
        var layers = luts.map { $0.apply(to: image) }
        while layers.count < 4 { layers.append(image) }  // grid channel is -1e4 -> weight 0
        let args: [Any] = [image] + layers + [
            gridImage,
            CIVector(x: e.minX, y: e.minY, z: e.width, w: e.height),
            CIVector(x: CGFloat(gridBins), y: CGFloat(gridSize.w), z: CGFloat(gridSize.h), w: 0),
        ]
        guard var out = kernel.apply(extent: e, roiCallback: { i, r in
            i == 5 ? gridImage.extent : r   // the grid is sampled anywhere; the rest per pixel
        }, arguments: args) else {
            throw NSError(domain: "GradeML", code: 11, userInfo: [NSLocalizedDescriptionKey: "localBlend failed"])
        }
        if intensity != 1 {
            let f = CIFilter.dissolveTransition()
            f.inputImage = image
            f.targetImage = out
            f.time = intensity
            out = f.outputImage ?? out
        }
        return out
    }
}

final class LocalGrader {
    let styles: [String]
    private let model: MLModel

    init(computeUnits: MLComputeUnits = .all) throws {
        guard let url = Bundle.main.url(forResource: "grade_local_lut", withExtension: "mlmodelc"),
              let stylesURL = Bundle.main.url(forResource: "grade_local_lut.styles", withExtension: "json") else {
            throw NSError(domain: "GradeML", code: 12, userInfo: [NSLocalizedDescriptionKey: "grade_local_lut model or styles missing"])
        }
        styles = try JSONDecoder().decode([String].self, from: Data(contentsOf: stylesURL))
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: config)
    }

    func predict(_ photo: CGImage, style name: String) throws -> LocalEdit {
        guard let k = styles.firstIndex(of: name) else {
            throw NSError(domain: "GradeML", code: 5, userInfo: [NSLocalizedDescriptionKey: "unknown style \(name)"])
        }
        let onehot = try MLMultiArray(shape: [1, NSNumber(value: styles.count)], dataType: .float32)
        for i in 0..<styles.count { onehot[i] = i == k ? 1 : 0 }
        let opts: [MLFeatureValue.ImageOption: Any] = [.cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue]
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "content": try MLFeatureValue(cgImage: photo, pixelsWide: 256, pixelsHigh: 256,
                                          pixelFormatType: kCVPixelFormatType_32BGRA, options: opts),
            "style": MLFeatureValue(multiArray: onehot),
        ])
        let out = try model.prediction(from: input)
        guard let luts = out.featureValue(for: "luts")?.multiArrayValue,
              let grid = out.featureValue(for: "grid")?.multiArrayValue else {
            throw NSError(domain: "GradeML", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing luts/grid output"])
        }
        return LocalGrader.edit(luts: luts, grid: grid)
    }

    /// luts (K, 3, D, D, D), grid (1, K, G, gh, gw) -> LocalEdit.
    static func edit(luts: MLMultiArray, grid: MLMultiArray) -> LocalEdit {
        let k = luts.shape[0].intValue, d = luts.shape[4].intValue
        var cubes: [ColorCube] = []
        for j in 0..<k {
            var data = [Float](repeating: 1, count: d * d * d * 4)
            for c in 0..<3 { for b in 0..<d { for g in 0..<d { for r in 0..<d {
                data[((b * d + g) * d + r) * 4 + c] = luts[[j, c, b, g, r] as [NSNumber]].floatValue
            } } } }
            cubes.append(ColorCube(size: d, data: data).clamped())
        }
        let bins = grid.shape[2].intValue, gh = grid.shape[3].intValue, gw = grid.shape[4].intValue
        return LocalEdit(luts: cubes, gridImage: pack(grid, k: k, bins: bins, gh: gh, gw: gw),
                         gridBins: bins, gridSize: (gw, gh))
    }

    /// Mirrors grademl.local.pack_grid: luma slices side by side, channel k = LUT k.
    /// Row 0 of the bitmap is the top of the photo, as grid row 0 is in PyTorch.
    static func pack(_ grid: MLMultiArray, k: Int, bins: Int, gh: Int, gw: Int) -> CIImage {
        let width = bins * gw
        var px = [Float](repeating: -1e4, count: width * gh * 4)
        for j in 0..<k { for z in 0..<bins { for y in 0..<gh { for x in 0..<gw {
            px[(y * width + z * gw + x) * 4 + j] = grid[[0, j, z, y, x] as [NSNumber]].floatValue
        } } } }
        let data = px.withUnsafeBufferPointer { Data(buffer: $0) }
        // No color space: the logits must reach the kernel untouched.
        return CIImage(bitmapData: data, bytesPerRow: width * 16, size: CGSize(width: width, height: gh),
                       format: .RGBAf, colorSpace: nil)
    }
}

/// MatchNet-L: content-aware matching (sky to sky, skin to skin) of a photo
/// to a reference, rendered like GradeNet-L. Bundle match_local_lut.mlmodelc
/// (scripts/export_coreml.py --model match_local).
///
///   let edit = try LocalMatcher().predict(photo: photo, reference: ref)
///   let out = try edit.render(CIImage(cgImage: photo), intensity: strength)
final class LocalMatcher {
    private let model: MLModel

    init(computeUnits: MLComputeUnits = .all) throws {
        guard let url = Bundle.main.url(forResource: "match_local_lut", withExtension: "mlmodelc") else {
            throw NSError(domain: "GradeML", code: 13, userInfo: [NSLocalizedDescriptionKey: "match_local_lut.mlmodelc missing"])
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: config)
    }

    func predict(photo: CGImage, reference: CGImage) throws -> LocalEdit {
        let opts: [MLFeatureValue.ImageOption: Any] = [.cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue]
        func feature(_ img: CGImage) throws -> MLFeatureValue {
            try MLFeatureValue(cgImage: img, pixelsWide: 256, pixelsHigh: 256,
                               pixelFormatType: kCVPixelFormatType_32BGRA, options: opts)
        }
        let out = try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: [
            "content": try feature(photo), "reference": try feature(reference),
        ]))
        guard let luts = out.featureValue(for: "luts")?.multiArrayValue,
              let grid = out.featureValue(for: "grid")?.multiArrayValue else {
            throw NSError(domain: "GradeML", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing luts/grid output"])
        }
        return LocalGrader.edit(luts: luts, grid: grid)
    }
}
