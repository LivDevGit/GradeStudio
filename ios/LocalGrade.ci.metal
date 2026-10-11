// LocalGrade.ci.metal — GradeNet-L / semantic match blend for Core Image.
//
// UNTESTED: written on Windows without a Metal toolchain. The math mirrors
// grademl.local.slice_packed, which tests/test_local.py checks against the
// training-time implementation (grademl.local.blend_weights).
//
// Build: add to the app target and set, for this file's target,
//   Other Metal Compiler Flags:  -fcikernel
//   Other Metal Linker Flags:    -cikernel
// (Xcode compiles it into default.metallib; LocalGrader loads "localBlend".)
//
// Inputs
//   src      the photo (any size)
//   l0..l3   the photo through each of the K LUTs (CIColorCube); unused
//            ones can be any image, their grid channel is -1e4 (weight 0)
//   grid     packed bilateral grid: (G * gw) x gh RGBA float image, luma
//            slice z in the z-th gw-wide tile, channel k = LUT k's logit
//   extent   (x, y, w, h) of the photo
//   dims     (G, gw, gh, 0)
//
// out = sum_k softmax_k(grid at (x, y, luma)) * l_k, blended on sRGB-encoded
// values like training (Core Image's working space is linear).

#include <CoreImage/CoreImage.h>

extern "C" {
namespace coreimage {

static float4 tap(sampler grid, float x, float y) {
    return grid.sample(grid.transform(float2(x, y)));
}

[[stitchable]] float4 localBlend(sampler src, sampler l0, sampler l1, sampler l2, sampler l3,
                                 sampler grid, float4 extent, float4 dims, destination dest) {
    float4 p = src.sample(src.coord());
    float3 s = linear_to_srgb(clamp(p.rgb, 0.0, 1.0));
    float luma = dot(s, float3(0.2126, 0.7152, 0.0722));

    float G = dims.x, gw = dims.y, gh = dims.z;
    float2 c = dest.coord() - extent.xy - 0.5;                 // pixel centers from 0
    float u = clamp(c.x / max(extent.z - 1.0, 1.0), 0.0, 1.0);
    float v = clamp(c.y / max(extent.w - 1.0, 1.0), 0.0, 1.0);
    // Core Image's y grows upward in both the photo and the grid image, so
    // v maps straight to the grid's rows.
    float gx = u * (gw - 1.0) + 0.5;
    float gy = v * (gh - 1.0) + 0.5;
    float z = luma * (G - 1.0);
    float z0 = clamp(floor(z), 0.0, G - 2.0);
    float fz = z - z0;
    float4 logit = mix(tap(grid, gx + z0 * gw, gy), tap(grid, gx + (z0 + 1.0) * gw, gy), fz);

    float4 e = exp(logit - max(max(logit.x, logit.y), max(logit.z, logit.w)));
    float4 w = e / (e.x + e.y + e.z + e.w);

    float3 out = w.x * linear_to_srgb(clamp(l0.sample(l0.coord()).rgb, 0.0, 1.0))
               + w.y * linear_to_srgb(clamp(l1.sample(l1.coord()).rgb, 0.0, 1.0))
               + w.z * linear_to_srgb(clamp(l2.sample(l2.coord()).rgb, 0.0, 1.0))
               + w.w * linear_to_srgb(clamp(l3.sample(l3.coord()).rgb, 0.0, 1.0));
    return float4(srgb_to_linear(out), p.a);
}

}
}
