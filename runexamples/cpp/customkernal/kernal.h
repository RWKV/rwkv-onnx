#define longint long
// simd
#ifdef __AVX512F__  // This macro is defined if AVX-512 is supported
  #include <immintrin.h>
  #define VALUE __m512
  #define SIMD_WIDTH 16
  #define LOAD(x) _mm512_load_ps(x)
  #define STORE(x, y) _mm512_store_ps(x, y)
  #define SET1(x) _mm512_set1_ps(x)
  #define MULTIPLY(x, y) _mm512_mul_ps(x, y)
  #define MULTADD(x, y, z) _mm512_fmadd_ps(x, y, z)
  // print out the SIMD width
    #pragma message("AVX-512 is supported")
#else
  // Fallback to AVX2 if AVX-512 is not supported
  #ifdef __AVX2__
    #include <immintrin.h>
    #define VALUE __m256
    #define SIMD_WIDTH 8
    #define LOAD(x) _mm256_load_ps(x)
    #define STORE(x, y) _mm256_store_ps(x, y)
    #define SET1(x) _mm256_set1_ps(x)
    #define MULTIPLY(x, y) _mm256_mul_ps(x, y)
    #define MULTADD(x, y, z) _mm256_fmadd_ps(x, y, z)
    // print out the SIMD width
    #pragma message("AVX-2 is supported")

  #else
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #include <arm_neon.h>
        #define VALUE float32x4_t
        #define SIMD_WIDTH 4  // NEON typically operates on 128-bit registers (4 floats)
        #define LOAD(x) vld1q_f32(x)
        #define STORE(x, y) vst1q_f32(x, y)
        #define SET1(x) vdupq_n_f32(x)
        #define MULTIPLY(x, y) vmulq_f32(x, y)
        #define MULTADD(x, y, z) vmlaq_f32(z, x, y)
        // Print out the SIMD width
        #pragma message("ARM NEON is supported")
    #else
        #pragma message("No SIMD is supported")
        #define VALUE float
        #define SIMD_WIDTH 1
        #define LOAD(x) (x)[0]
        #define STORE(x, y) (x)[0] = y
        #define SET1(x) x
        #define MULTIPLY(x, y) x * y
        #define MULTADD(x, y, z) x * y + z
    #endif

    #endif

#endif


void forward_cpu(longint B, longint T, longint C, longint H, float* ss, float* rr, float* kk, float* vv, float* ww, float* uu, float* out) {
    
    

    // 1d tensor
    longint tsize = B*H*(C/H);
    // 2d tensor
    longint ttsize = B*H*(C/H)*(C/H);

    // 1d 
    longint bsize = H*(C/H);
    // 2d
    longint bbsize = H*(C/H)*(C/H);

    // 1d
    longint hsize = (C/H);
    // 2d
    longint hhsize = (C/H)*(C/H);

    for (longint t = 0; t < T; t++) {

        longint timeoffset = t * tsize;

        for (longint bb = 0; bb < B; bb++) {
            

            longint btimeoffset = timeoffset + bb * bsize;
            longint bbhsize = bb * bbsize;

            for (longint hh = 0; hh < H; hh++) {
                longint hoffset = hh * hsize;
                longint bhofseti = btimeoffset + hoffset;
                longint bbhhsize = bbhsize + hh * hhsize;

                for (longint i = 0; i < C/H; i++) {

                    longint iind = bhofseti + i;
                    longint hoffseti = hoffset + i;  
                    longint bbhhofseti = bbhhsize + i * hsize;  
                   
                    //auto kkk = kk[iind];
                    VALUE kkk = SET1(kk[iind]);  
                    VALUE uuu = SET1(uu[hoffseti]); 
                    VALUE rrr = SET1(rr[iind]);
                    VALUE www = SET1(ww[hoffseti]);

                   


                    for (longint j = 0; j < C/H; j+=SIMD_WIDTH) {
                        longint jind = bhofseti + j;
                        longint sind = bbhhofseti + j;

                        
                        
                        // atu = k[t,bb,hh,i]*v[t,bb,hh,j]
                        VALUE vvv = LOAD(&vv[jind]);

                        // multiply kkk and vvv
                        VALUE atu = MULTIPLY(vvv,kkk);

                        VALUE sss = LOAD(&ss[sind]);

                        // out[t,bb,hh,j] += r[t,bb,hh,i]*(s[bb,hh,i,j] + atu*u[hh,i] )
                        VALUE sssatuuuu = MULTADD(atu,uuu,sss);

                        VALUE outtt = LOAD(&out[jind]);

                        printf("Batch: %ld\n", i);

                        STORE(&out[jind], MULTADD(sssatuuuu,rrr,outtt));

                        // s[bb,hh,i,j] = s[bb,hh,i,j] * w[hh,i] + atu
                        STORE(&ss[sind], MULTADD(sss,www,atu));
                    }

                }
            }
        }
    }

    
}



// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("forward_cpu", &forward_cpu, "CPU forward");
// }

// TORCH_LIBRARY(wkv5, m) {
//     m.def("forward_cpu", forward_cpu);
// }

