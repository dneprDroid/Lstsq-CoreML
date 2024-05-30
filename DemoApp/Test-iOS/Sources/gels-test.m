
#import <Foundation/Foundation.h>
//#import <Accelerate/Accelerate.h>
#import <math.h>
#include <dlfcn.h>

/* SGELS prototype

 nrhs = b

 libtorch_cpu.dylib
 */

//extern void sgelsd_(int *m, int *n, int *nrhs,
//    float *a, int *lda, float *b, int *ldb,
//    float *s, float *rcond, int *rank,
//    float *work, int *lwork, int *iwork, int *info);
void sgels_( char* trans, int* m, int* n, int* nrhs, float* a, int* lda,
                float* b, int* ldb, float* work, int* lwork, int* info );
///* Auxiliary routines prototypes */
extern void print_matrix( char* desc, int m, int n, float* a, int lda );
extern void print_vector_norm( char* desc, int m, int n, float* a, int lda );

/* Parameters */
#define M 3
#define N 5
#define NRHS 1
#define LDA M
#define LDB (M > N ? M : N)

//void *jjjj = sgels_;
/* Main program */
int gels_test() {
//        const char *libPath = "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/torch/lib/libtorch_cpu.dylib";
        
         void (*sgels_func)(
                          char* trans, int* m, int* n, int* nrhs, float* a, int* lda,
                                         float* b, int* ldb, float* work, int* lwork, int* info
         );

//         void *handle = dlopen(libPath, RTLD_NOW);
//        
//         if (!handle) {
//             printf("Can't find framework\n");
//             return 1;
//         }
//         printf("Found framework\n");

//         *(void**)(&sgels_func) = dlsym(handle, "sgels_");
    *(void**)(&sgels_func) = sgels_;
    
        /* Locals */
        int m = M, n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info, lwork;
        float wkopt;
        float* work;
        /* Local arrays */
        float a[LDA*N] = {
                        1, 6, 11,
                        2, 7, 12,
                        3, 8, 13,
                        4, 9, 14,
                        5, 10, 15
        };
        float b[LDB*NRHS] = {
            355, 930, 1505
        };
        /* Executable statements */
        printf( " SGELS Example Program Results\n" );
        /* Query and allocate the optimal workspace */
        lwork = -1;
    sgels_func( "N", &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork,
                        &info );
        lwork = (int)wkopt;
        work = (float*)malloc( lwork*sizeof(float) );
        /* Solve the equations A*X = B */
    sgels_func( "N", &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
                        &info );
        /* Check for the full rank */
        if( info > 0 ) {
                printf( "The diagonal element %i of the triangular factor ", info );
                printf( "of A is zero, so that A does not have full rank;\n" );
                printf( "the least squares solution could not be computed.\n" );
                exit( 1 );
        }
        /* Print least squares solution */
        print_matrix( "Least squares solution", n, nrhs, b, ldb );
        /* Print residual sum of squares for the solution */
        print_vector_norm( "Residual sum of squares for the solution", m-n, nrhs,
                        &b[n], ldb );
        /* Print details of QR factorization */
        print_matrix( "Details of QR factorization", m, n, a, lda );
        /* Free workspace */
        free( (void*)work );
//        exit( 0 );
    return 0;
} /* End of SGELS Example */

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, int m, int n, float* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
                printf( "\n" );
        }
}

/* Auxiliary routine: printing norms of matrix columns */
void print_vector_norm( char* desc, int m, int n, float* a, int lda ) {
        int i, j;
        float norm;
        printf( "\n %s\n", desc );
        for( j = 0; j < n; j++ ) {
                norm = 0.0;
                for( i = 0; i < m; i++ ) norm += a[i+j*lda] * a[i+j*lda];
                printf( " %6.2f", norm );
        }
        printf( "\n" );
}
