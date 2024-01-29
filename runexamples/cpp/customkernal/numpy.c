#define longint long
// simd
#ifdef __AVX512F__  // This macro is defined if AVX-512 is supported
  #include <immintrin.h>
  #define VALUE __m512
  #define SIMD_WIDTH 16
  #define LOAD(x) _mm512_loadu_ps(x)
  #define STORE(x, y) _mm512_storeu_ps(x, y)
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

                        STORE(&out[jind], MULTADD(sssatuuuu,rrr,outtt));

                        // s[bb,hh,i,j] = s[bb,hh,i,j] * w[hh,i] + atu
                        STORE(&ss[sind], MULTADD(sss,www,atu));
                    }

                }
            }
        }
    }

    
}


#include <stdio.h>
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

// Module method definitions
static PyObject* wkvworld_c(PyObject *self, PyObject *args) {
    
    Py_RETURN_NONE;
}


static PyObject* wkvnumpy_c(PyObject *dummy, PyObject *args)
{
    
    PyObject *KK=NULL;
    PyObject *VV=NULL;
    PyObject *RR=NULL;
    PyObject *TD=NULL;
    PyObject *TF=NULL;
    PyObject *STATE=NULL;
 
    int nd;

    if (!PyArg_ParseTuple(args, "OOOOOO", &KK, &VV, &RR, &TD, &TF, &STATE))
        return NULL;

    /*
     * my code starts here
     */

    npy_intp *sp=PyArray_SHAPE(STATE);

    long Batch = sp[0];
    long Time = 1;
    long Head = sp[1];
    long Headsize = sp[2];
    long Channel = Head * Headsize;

    // asume rank 4
    // printf("array dimentsion: %ld %ld %ld %ld\n",*sp, *(sp+1), *(sp+2), *(sp+3));

    float* ss = (float*)PyArray_DATA(STATE);
    float* rr = (float*)PyArray_DATA(RR);
    float* kk = (float*)PyArray_DATA(KK);
    float* vv = (float*)PyArray_DATA(VV);
    float* td = (float*)PyArray_DATA(TD);
    float* tf = (float*)PyArray_DATA(TF);

    // allocate output
    npy_intp out_dims[4] = {Batch, Head, 1, Headsize};
    PyObject *OUT = PyArray_SimpleNew(4, out_dims, NPY_FLOAT);
    float* out = (float*)PyArray_DATA(OUT);
    // fill with 0
    for (int i = 0; i < Batch * Head * Headsize; i+=SIMD_WIDTH) {
        STORE(&out[i], SET1(0));
    }

    // allocate new STATE, copy from old one
    npy_intp state_dims[4] = {Batch, Head, Headsize, Headsize};
    PyObject *STATE_NEW = PyArray_SimpleNew(4, state_dims, NPY_FLOAT);
    // get mutable pointer
    float* ss_new = (float*)PyArray_DATA(STATE_NEW);

    for (int i = 0; i < Batch * Head * Headsize * Headsize; i+=SIMD_WIDTH) {
        STORE(&ss_new[i], LOAD(&ss[i]));
    }

    

    


    forward_cpu(Batch, Time, Channel, Head, ss_new, rr, kk, vv, td, tf, out);
    // ss_new[0] = 0.9;
    /// return touple of both output and new state
    
    PyObject *ret = PyTuple_New(2);
    PyTuple_SetItem(ret, 0, OUT);
    PyTuple_SetItem(ret, 1, STATE_NEW);
    return ret;
}


static PyMethodDef methods[] = {
        {
                "wkvpython", wkvworld_c, METH_VARARGS,
                "Print 'hello xxx'"
        },
        {
                "wkv5", wkvnumpy_c, METH_VARARGS,
                "numpy function tester", 6
                
        },
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef wkvdefinition = {
        PyModuleDef_HEAD_INIT,
        "wkv5",
        "A Python module that allows for fast wkv5 operation",
        -1,
        methods
};


PyMODINIT_FUNC PyInit_wkv5(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&wkvdefinition);
}