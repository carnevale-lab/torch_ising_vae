/*
 * AUTOGENERATED DON'T EDIT
 * Please make changes to the code generator (distutils/ccompiler_opt.py)
*/
#define NPY_WITH_CPU_BASELINE  "SSE SSE2 SSE3"
#define NPY_WITH_CPU_DISPATCH  "SSSE3 SSE41 POPCNT SSE42 AVX F16C FMA3 AVX2"
#define NPY_WITH_CPU_BASELINE_N 3
#define NPY_WITH_CPU_DISPATCH_N 8
#define NPY_WITH_CPU_EXPAND_(X) X
#define NPY_WITH_CPU_BASELINE_CALL(MACRO_TO_CALL, ...) \
	NPY_WITH_CPU_EXPAND_(MACRO_TO_CALL(SSE, __VA_ARGS__)) \
	NPY_WITH_CPU_EXPAND_(MACRO_TO_CALL(SSE2, __VA_ARGS__)) \
	NPY_WITH_CPU_EXPAND_(MACRO_TO_CALL(SSE3, __VA_ARGS__))
#define NPY_WITH_CPU_DISPATCH_CALL(MACRO_TO_CALL, ...) \
	NPY_WITH_CPU_EXPAND_(MACRO_TO_CALL(SSSE3, __VA_ARGS__)) \
	NPY_WITH_CPU_EXPAND_(MACRO_TO_CALL(SSE41, __VA_ARGS__)) \
	NPY_WITH_CPU_EXPAND_(MACRO_TO_CALL(POPCNT, __VA_ARGS__)) \
	NPY_WITH_CPU_EXPAND_(MACRO_TO_CALL(SSE42, __VA_ARGS__)) \
	NPY_WITH_CPU_EXPAND_(MACRO_TO_CALL(AVX, __VA_ARGS__)) \
	NPY_WITH_CPU_EXPAND_(MACRO_TO_CALL(F16C, __VA_ARGS__)) \
	NPY_WITH_CPU_EXPAND_(MACRO_TO_CALL(FMA3, __VA_ARGS__)) \
	NPY_WITH_CPU_EXPAND_(MACRO_TO_CALL(AVX2, __VA_ARGS__))
/******* baseline features *******/
	/** SSE **/
	#define NPY_HAVE_SSE 1
	#include <xmmintrin.h>
	/** SSE2 **/
	#define NPY_HAVE_SSE2 1
	#include <emmintrin.h>
	/** SSE3 **/
	#define NPY_HAVE_SSE3 1
	#include <pmmintrin.h>

/******* dispatch features *******/
#ifdef NPY__CPU_TARGET_SSSE3
	/** SSSE3 **/
	#define NPY_HAVE_SSSE3 1
	#include <tmmintrin.h>
#endif /*NPY__CPU_TARGET_SSSE3*/
#ifdef NPY__CPU_TARGET_SSE41
	/** SSE41 **/
	#define NPY_HAVE_SSE41 1
	#include <smmintrin.h>
#endif /*NPY__CPU_TARGET_SSE41*/
#ifdef NPY__CPU_TARGET_POPCNT
	/** POPCNT **/
	#define NPY_HAVE_POPCNT 1
	#include <popcntintrin.h>
#endif /*NPY__CPU_TARGET_POPCNT*/
#ifdef NPY__CPU_TARGET_SSE42
	/** SSE42 **/
	#define NPY_HAVE_SSE42 1
#endif /*NPY__CPU_TARGET_SSE42*/
#ifdef NPY__CPU_TARGET_AVX
	/** AVX **/
	#define NPY_HAVE_AVX 1
	#include <immintrin.h>
#endif /*NPY__CPU_TARGET_AVX*/
#ifdef NPY__CPU_TARGET_F16C
	/** F16C **/
	#define NPY_HAVE_F16C 1
#endif /*NPY__CPU_TARGET_F16C*/
#ifdef NPY__CPU_TARGET_FMA3
	/** FMA3 **/
	#define NPY_HAVE_FMA3 1
#endif /*NPY__CPU_TARGET_FMA3*/
#ifdef NPY__CPU_TARGET_AVX2
	/** AVX2 **/
	#define NPY_HAVE_AVX2 1
#endif /*NPY__CPU_TARGET_AVX2*/
