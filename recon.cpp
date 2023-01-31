

#include "YAKL.h"
using yakl::SArray;
typedef float real;
YAKL_INLINE real constexpr operator"" _fp( long double x ) {
  return static_cast<real>(x);
}
typedef yakl::Array<real,1,yakl::memDevice,yakl::styleC> real1d;
typedef yakl::Array<real,2,yakl::memDevice,yakl::styleC> real2d;
typedef yakl::Array<real,3,yakl::memDevice,yakl::styleC> real3d;
typedef yakl::Array<real,4,yakl::memDevice,yakl::styleC> real4d;
typedef yakl::Array<real,5,yakl::memDevice,yakl::styleC> real5d;
typedef yakl::Array<real,6,yakl::memDevice,yakl::styleC> real6d;
typedef yakl::Array<real,7,yakl::memDevice,yakl::styleC> real7d;
typedef yakl::Array<real const,1,yakl::memDevice,yakl::styleC> realConst1d;
typedef yakl::Array<real const,2,yakl::memDevice,yakl::styleC> realConst2d;
typedef yakl::Array<real const,3,yakl::memDevice,yakl::styleC> realConst3d;
typedef yakl::Array<real const,4,yakl::memDevice,yakl::styleC> realConst4d;
typedef yakl::Array<real const,5,yakl::memDevice,yakl::styleC> realConst5d;
typedef yakl::Array<real const,6,yakl::memDevice,yakl::styleC> realConst6d;
typedef yakl::Array<real const,7,yakl::memDevice,yakl::styleC> realConst7d;
#include "TransformMatrices.h"
#include "WenoLimiter.h"

#ifdef ORD
#define ord ORD
#else
#define ord 3
#endif
int constexpr hs = (ord-1)/2;

YAKL_INLINE void reconstruct_gll_values( SArray<real,1,ord> const stencil                      ,
                                         SArray<real,1,2> &gll                                 ,
                                         SArray<real,2,ord,2> const &coefs_to_gll              ,
                                         SArray<real,2,ord,ord>  const &sten_to_coefs          ,
                                         SArray<real,3,hs+1,hs+1,hs+1> const &weno_recon_lower ,
                                         SArray<real,1,hs+2> const &idl                        ,
                                         real sigma );


int main(int argc, char **argv) {
  yakl::init();
  {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;

    int constexpr num_state = 5;
    int nx = 1024;
    int ny = 1024;
    int nz = 4;

    SArray<real,2,ord,ord>        sten_to_coefs;    // Matrix to convert ord stencil avgs to ord poly coefs
    SArray<real,2,ord,2  >        coefs_to_gll;     // Matrix to convert ord poly coefs to two GLL points
    SArray<real,3,hs+1,hs+1,hs+1> weno_recon_lower; // WENO's lower-order reconstruction matrices (sten_to_coefs)
    SArray<real,1,hs+2>           weno_idl;         // Ideal weights for WENO
    real                          weno_sigma;       // WENO sigma parameter (handicap high-order TV estimate)
    TransformMatrices::sten_to_coefs           (sten_to_coefs   );
    TransformMatrices::coefs_to_gll_lower      (coefs_to_gll    );
    TransformMatrices::weno_lower_sten_to_coefs(weno_recon_lower);
    weno::wenoSetIdealSigma<ord>( weno_idl , weno_sigma );

    real4d state_init("state",num_state,nz,ny,nx);
    state_init = 1;
    realConst4d state = state_init;

    // These arrays store high-order-accurate samples of the state at cell edges after cell-centered recon
    real5d state_limits_x("state_limits_x",num_state,2,nz  ,ny  ,nx+1);
    real5d state_limits_y("state_limits_y",num_state,2,nz  ,ny+1,nx  );
    real5d state_limits_z("state_limits_z",num_state,2,nz+1,ny  ,nx  );

    // Compute samples of state at cell edges using cell-centered reconstructions at high-order with WENO
    // At the end of this, we will have two samples per cell edge in each dimension, one from each adjacent cell.
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i ) {
      // X-direction
      {
        SArray<real,1,ord> stencil;
        SArray<real,1,2>   gll;
        for (int s=0; s < ord; s++) { stencil(s) = state(l,hs+k,hs+j,i+s); }
        reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
        state_limits_x(l,1,k,j,i  ) = gll(0);
        state_limits_x(l,0,k,j,i+1) = gll(1);
      }

      // Y-direction
      {
        SArray<real,1,ord> stencil;
        SArray<real,1,2>   gll;
        for (int s=0; s < ord; s++) { stencil(s) = state(l,hs+k,j+s,hs+i); }
        reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
        state_limits_y(l,1,k,j  ,i) = gll(0);
        state_limits_y(l,0,k,j+1,i) = gll(1);
      }

      // Z-direction
      {
        SArray<real,1,ord> stencil;
        SArray<real,1,2>   gll;
        for (int s=0; s < ord; s++) {
          int ind_k = k+s;
          if (ind_k < hs     ) ind_k += nz;
          if (ind_k > hs+nz-1) ind_k -= nz;
          stencil(s) = state(l,ind_k,hs+j,hs+i);
        }
        reconstruct_gll_values(stencil,gll,coefs_to_gll,sten_to_coefs,weno_recon_lower,weno_idl,weno_sigma);
        state_limits_z(l,1,k  ,j,i) = gll(0);
        state_limits_z(l,0,k+1,j,i) = gll(1);
      }
    });
  }
  yakl::finalize();
}


YAKL_INLINE void reconstruct_gll_values( SArray<real,1,ord> const stencil                      ,
                                         SArray<real,1,2> &gll                                 ,
                                         SArray<real,2,ord,2> const &coefs_to_gll              ,
                                         SArray<real,2,ord,ord>  const &sten_to_coefs          ,
                                         SArray<real,3,hs+1,hs+1,hs+1> const &weno_recon_lower ,
                                         SArray<real,1,hs+2> const &idl                        ,
                                         real sigma ) {
  // Reconstruct values
  SArray<real,1,ord> wenoCoefs;
  weno::compute_weno_coefs<ord>( weno_recon_lower , sten_to_coefs , stencil , wenoCoefs , idl , sigma );
  // Transform ord weno coefficients into 2 GLL points
  for (int ii=0; ii<2; ii++) {
    real tmp = 0;
    for (int s=0; s < ord; s++) {
      tmp += coefs_to_gll(s,ii) * wenoCoefs(s);
    }
    gll(ii) = tmp;
  }
}


