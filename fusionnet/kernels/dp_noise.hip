/*
 * FusionNet: HIP Kernel for Differential Privacy Noise Addition
 * =============================================================
 * Implements constant-time Gaussian noise addition for DP-SGD on AMD GPUs.
 *
 * Algorithm:
 *   1. Per-element gradient clipping: grad = clamp(grad, -clip_norm, +clip_norm)
 *   2. Gaussian noise sampling:       noise ~ N(0, sigma^2)
 *   3. Noisy gradient:                noisy_grad = grad + noise
 *
 * where sigma = (clip_norm * sqrt(2 * ln(1.25/delta))) / epsilon
 *
 * Compile with:
 *   hipcc -O3 -o dp_noise_test dp_noise.hip -lhiprand
 *
 * Requirements:
 *   ROCm >= 5.0, hipRAND library (included in ROCm stack)
 */

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <math.h>

/* ───────────────────────────────────────────────────────────────
 * Kernel 1: RNG State Initialisation
 * Each thread gets its own Philox4_32_10 PRNG state, seeded with
 * (global_seed, thread_index). Philox is AMD's recommended CSPRNG
 * for cryptographically adequate noise in DP workloads.
 * ─────────────────────────────────────────────────────────────── */
__global__ void dp_sgd_setup_kernel(
    hiprandStatePhilox4_32_10_t* __restrict__ states,
    unsigned long long seed,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    /* Each thread uses its index as the sequence number so states
     * are statistically independent across elements.             */
    hiprand_init(seed, (unsigned long long)idx, 0ULL, &states[idx]);
}

/* ───────────────────────────────────────────────────────────────
 * Kernel 2: Gradient Clipping + Gaussian Noise Addition
 * Operates in-place on `gradients` array.
 *
 * Args:
 *   gradients  — float array of gradient values (modified in-place)
 *   states     — per-thread hiprand PRNG states (from setup kernel)
 *   sigma      — noise standard deviation  (= clip_norm * noise_multiplier)
 *   clip_norm  — per-element gradient clipping bound C
 *   num_elements — length of the gradients array
 * ─────────────────────────────────────────────────────────────── */
__global__ void dp_sgd_noise_kernel(
    float* __restrict__ gradients,
    hiprandStatePhilox4_32_10_t* __restrict__ states,
    float sigma,
    float clip_norm,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    /* Step 1: Element-wise gradient clipping */
    float grad = gradients[idx];
    grad = fmaxf(-clip_norm, fminf(clip_norm, grad));

    /* Step 2: Sample N(0, sigma^2) Gaussian noise
     * hiprand_normal returns N(0,1); multiply by sigma for N(0,sigma^2). */
    float noise = hiprand_normal(&states[idx]) * sigma;

    /* Step 3: Write noisy clipped gradient */
    gradients[idx] = grad + noise;
}

/* ───────────────────────────────────────────────────────────────
 * Host-side Launcher
 * Wraps both kernels into a single call. Manages RNG state
 * allocation and launch configuration automatically.
 *
 * Args:
 *   d_gradients   — device pointer to gradient buffer
 *   num_elements  — number of gradient elements
 *   epsilon       — DP privacy budget (ε)
 *   delta         — DP failure probability (δ)
 *   clip_norm     — gradient clipping norm (C)
 *   seed          — RNG seed (use round_num XOR client_id for variety)
 * ─────────────────────────────────────────────────────────────── */
void fusionnet_dp_noise(
    float* d_gradients,
    int num_elements,
    float epsilon,
    float delta,
    float clip_norm,
    unsigned long long seed
) {
    /* Compute noise multiplier: σ = C * sqrt(2 * ln(1.25/δ)) / ε */
    float sigma = clip_norm * sqrtf(2.0f * logf(1.25f / delta)) / epsilon;

    const int BLOCK_SIZE = 256;
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Allocate per-thread RNG states on device */
    hiprandStatePhilox4_32_10_t* d_states;
    hipMalloc(&d_states, num_elements * sizeof(hiprandStatePhilox4_32_10_t));

    /* Launch 1: Initialise RNG states */
    hipLaunchKernelGGL(
        dp_sgd_setup_kernel,
        dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
        d_states, seed, num_elements
    );

    /* Launch 2: Clip gradients + add Gaussian noise */
    hipLaunchKernelGGL(
        dp_sgd_noise_kernel,
        dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
        d_gradients, d_states, sigma, clip_norm, num_elements
    );

    hipFree(d_states);
    hipDeviceSynchronize();
}

/* ───────────────────────────────────────────────────────────────
 * Standalone test (compile and run directly for benchmarking)
 * hipcc -O3 -o dp_noise_test dp_noise.hip -lhiprand && ./dp_noise_test
 * ─────────────────────────────────────────────────────────────── */
#ifdef DP_NOISE_STANDALONE_TEST
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int N = 1 << 20;   /* 1M gradient elements */
    const float EPSILON    = 1.0f;
    const float DELTA      = 1e-5f;
    const float CLIP_NORM  = 1.0f;

    /* Allocate and fill host gradients with 1.0 */
    float* h_gradients = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) h_gradients[i] = 1.0f;

    /* Copy to device */
    float* d_gradients;
    hipMalloc(&d_gradients, N * sizeof(float));
    hipMemcpy(d_gradients, h_gradients, N * sizeof(float), hipMemcpyHostToDevice);

    /* Run DP noise kernel */
    fusionnet_dp_noise(d_gradients, N, EPSILON, DELTA, CLIP_NORM, /*seed=*/42ULL);

    /* Copy back and print stats */
    hipMemcpy(h_gradients, d_gradients, N * sizeof(float), hipMemcpyDeviceToHost);

    float sum = 0.0f, sq_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum    += h_gradients[i];
        sq_sum += h_gradients[i] * h_gradients[i];
    }
    float mean = sum / N;
    float std  = sqrtf(sq_sum / N - mean * mean);

    /* Expected: mean ~ 1.0 (clipped), std ~ sigma = sqrt(2*ln(1.25/1e-5)) ~ 3.05 */
    printf("DP Noise Test (N=%d, ε=%.1f, δ=%.0e, C=%.1f)\n", N, EPSILON, DELTA, CLIP_NORM);
    printf("  Mean : %.4f  (expected ~1.0 for all-ones input after clipping)\n", mean);
    printf("  Std  : %.4f  (expected ~%.4f for σ = C*sqrt(2*ln(1.25/δ))/ε)\n",
           std, CLIP_NORM * sqrtf(2.0f * logf(1.25f / DELTA)) / EPSILON);

    hipFree(d_gradients);
    free(h_gradients);
    return 0;
}
#endif /* DP_NOISE_STANDALONE_TEST */
