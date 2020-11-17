#include <iostream>
#include <hlib.hh>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>


namespace py = pybind11;

using namespace std;
using namespace HLIB;

using real_t = HLIB::real;

typedef std::vector< idx_t > IndexVector;
typedef py::array_t<real_t> PythonArray;
typedef std::function<PythonArray(IndexVector, IndexVector)> SubmatrixFunction;

// std::function<float(int, int)> func;
  
// CoeffFnFromPython(hilbert_submatrix)

class CoeffFnFromPython : public TCoeffFn< real_t > {
private:
    const std::shared_ptr<SubmatrixFunction*> compute_submatrix;
public:
    CoeffFnFromPython ( std::shared_ptr<SubmatrixFunction*> compute_submatrix )
            : compute_submatrix(compute_submatrix)
    {}
       
    virtual void eval(const IndexVector &  rowidxs, 
                      const IndexVector &  colidxs, 
                      real_t *             ptr_submatrix) const {
        
        int nrow = rowidxs.size();
        int ncol = colidxs.size();
        cout << "nrow=" << nrow << ", ncol=" << ncol << "\n" << endl << flush;
        fflush(stdout);

        // pybind11::gil_scoped_release release;
        PythonArray submatrix_numpy = (*compute_submatrix)(rowidxs, colidxs);
        // pybind11::gil_scoped_acquire acquire;

        py::buffer_info submatrix_buf(submatrix_numpy.request());
        real_t *ptr_submatrix2 = static_cast<real_t *>(submatrix_buf.ptr);

        cout << ptr_submatrix2[0] << " , " << ptr_submatrix2[nrow*ncol-1] << "\n" << endl << flush;
        fflush(stdout);



        // if (submatrix_numpy.size() > 0) {
            // for (size_t idx = 0; idx < submatrix_numpy.size(); idx++)
            // {
            //     // cout << "idx=" << idx << ", ptr_submatrix2[idx]=" << ptr_submatrix2[idx] << "\n" << endl << flush;
            //     // ptr_submatrix[idx] = ptr_submatrix2[idx];
            //     ptr_submatrix[idx] = 1.4;
            // }
        // }

        for (size_t idx = 0; idx < nrow * ncol; idx++)
        {
            // cout << "idx=" << idx << ", ptr_submatrix2[idx]=" << ptr_submatrix2[idx] << "\n" << endl << flush;
            // ptr_submatrix[idx] = ptr_submatrix2[idx];
            ptr_submatrix[idx] = 1.4;
        }
    }
    using TCoeffFn< real_t >::eval;
 
    virtual matform_t  matrix_format  () const { return symmetric; }
 
    virtual bool       is_complex     () const { return false; }

    void eval_from_python(IndexVector & rowidxs, 
                          IndexVector & colidxs, 
                          PythonArray & submatrix_numpy)
         {
            py::buffer_info submatrix_buf = submatrix_numpy.request();
            real_t *ptr_submatrix = static_cast<real_t *>(submatrix_buf.ptr);
            eval(rowidxs, colidxs, ptr_submatrix);
         }
};


// CoeffFnFromPython make_coeff_from_python(SubmatrixFunction f) {
//     return CoeffFnFromPython(f);
// }


class TLogCoeffFn : public TCoeffFn< real_t > {
private:
    const double  _h;
 
public:
    TLogCoeffFn ( const double  h )
            : _h(h)
    {}
 
  virtual void eval  ( const std::vector< idx_t > &  rowidxs,
                       const std::vector< idx_t > &  colidxs,
                       real_t *                      matrix ) const
  {
        const size_t  n = rowidxs.size();
        const size_t  m = colidxs.size();
 
        for ( size_t  j = 0; j < m; ++j )
        {
            const int  idx1 = colidxs[ j ];
            
            for ( size_t  i = 0; i < n; ++i )
            {
                const int  idx0 = rowidxs[ i ];
                double     value;
 
                if ( idx0 == idx1 ) 
                    value = -1.5*_h*_h + _h*_h*std::log(_h);
                else
                {
                    const double dist = _h * ( std::abs( idx0 - idx1 ) - 1 );
                    const double t1   = dist+1.0*_h;
                    const double t2   = dist+2.0*_h;
            
                    value = ( - 1.5*_h*_h + 0.5*t2*t2*std::log(t2)
                              - t1*t1*std::log(t1) );
            
                    if ( std::abs(dist) > 1e-8 )
                        value += 0.5*dist*dist*std::log(dist);
                }
        
                matrix[ j*n + i ] = -value;
            }
        }
    }
    using TCoeffFn< real_t >::eval;
 
    virtual matform_t  matrix_format  () const { return symmetric; }
 
    virtual bool       is_complex     () const { return false; }
};
 
real_t rhs ( const idx_t  i,
           const idx_t  n )
{
    const real_t  a     = real_t(i)             / real_t(n);
    const real_t  b     = (real_t(i)+real_t(1)) / real_t(n);
    real_t        value = -1.5 * (b - a);
  
    if ( std::abs( b )       > 1e-8 ) value += 0.5*b*b*std::log(b);
    if ( std::abs( a )       > 1e-8 ) value -= 0.5*a*a*std::log(a);
    if ( std::abs( 1.0 - b ) > 1e-8 ) value -= 0.5*(1.0-b)*(1.0-b)*
                                               std::log(1.0-b);
    if ( std::abs( 1.0 - a ) > 1e-8 ) value += 0.5*(1.0-a)*(1.0-a)*
                                               std::log(1.0-a); 
  
    return value;
}
 
int main ( int argc, char ** argv ) 
{
  try
  {
    INIT();
    CFG::set_verbosity( 3 );
 
    const size_t              n = 1024;
    const double              h = 1.0 / double(n);
    std::vector< double * >   vertices( n );
 
    for ( size_t i = 0; i < n; i++ )
    {
        vertices[i]    = new double;
        vertices[i][0] = h * double(i) + ( h / 2.0 ); // center of [i/h,(i+1)/h]
    }
 
    auto                      coord = make_unique< TCoordinate >( vertices, 1 );
 
    TAutoBSPPartStrat         part_strat;
    TBSPCTBuilder             ct_builder( & part_strat, 20 );
    auto                      ct = ct_builder.build( coord.get() );
 
    TStdGeomAdmCond           adm_cond( 2.0 );
    TBCBuilder                bct_builder;
    auto                      bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );
 
    TLogCoeffFn               log_coefffn( h );
    TPermCoeffFn< real_t >    coefffn( & log_coefffn, ct->perm_i2e(), ct->perm_i2e() );
    TACAPlus< real_t >        aca( & coefffn );
    TDenseMBuilder< real_t >  h_builder( & coefffn, & aca );
    TTruncAcc                 acc( 1e-6, 0.0 );
    auto                      A = h_builder.build( bct.get(), acc );
 
    auto                      B( A->copy() );
    auto                      A_inv = factorise_inv( B.get(), acc );
 
    auto                      b = A->row_vector();
 
    for ( size_t  i = 0; i < n; i++ )
        b->set_entry( i, rhs( i, n ) );
 
    ct->perm_e2i()->permute( b.get() );
 
    TAutoSolver               solver( 1000 );
    TSolverInfo               solve_info;
    auto                      x = A->col_vector();
 
    solver.solve( A.get(), x.get(), b.get(), A_inv.get(), & solve_info );
 
    ct->perm_i2e()->permute( x.get() );
 
    auto                      sol = b->copy();
 
    sol->fill( 1.0 );
    x->axpy( 1.0, sol.get() );
    
    std::cout << "  |x-x~| = " << x->norm2() << std::endl;
 
    DONE();
  }
  catch ( Error & e )
  {
    std::cout << e.to_string() << std::endl;
  }
 
  return 0;
}


// void run_problem (CoeffFnFromPython & log_coefffn) 
void run_problem (SubmatrixFunction f) 
{
  cout << "asdf" << "\n" << endl << flush;
  try
  {
    INIT();
    CFG::set_verbosity( 3 );
 
    const size_t              n = 1024;
    const double              h = 1.0 / double(n);
    std::vector< double * >   vertices( n );
 
    for ( size_t i = 0; i < n; i++ )
    {
        vertices[i]    = new double;
        vertices[i][0] = h * double(i) + ( h / 2.0 ); // center of [i/h,(i+1)/h]
    }
 
    auto                      coord = make_unique< TCoordinate >( vertices, 1 );
 
    TAutoBSPPartStrat         part_strat;
    TBSPCTBuilder             ct_builder( & part_strat, 20 );
    auto                      ct = ct_builder.build( coord.get() );
 
    TStdGeomAdmCond           adm_cond( 2.0 );
    TBCBuilder                bct_builder;
    auto                      bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );
 
    // TLogCoeffFn               log_coefffn( h );
    CoeffFnFromPython         log_coefffn(f);
    TPermCoeffFn< real_t >    coefffn( & log_coefffn, ct->perm_i2e(), ct->perm_i2e() );
    TACAPlus< real_t >        aca( & coefffn );
    TDenseMBuilder< real_t >  h_builder( & coefffn, & aca );
    TTruncAcc                 acc( 1e-6, 0.0 );
    
    cout << "eeee" << endl << flush;
    auto                      A = h_builder.build( bct.get(), acc );
    cout << "QQQQ" << endl << flush;
    fflush(stdout);
 
    auto                      B( A->copy() );
    auto                      A_inv = factorise_inv( B.get(), acc );
 
    auto                      b = A->row_vector();
 
    for ( size_t  i = 0; i < n; i++ )
        b->set_entry( i, rhs( i, n ) );
 
    ct->perm_e2i()->permute( b.get() );
 
    TAutoSolver               solver( 1000 );
    TSolverInfo               solve_info;
    auto                      x = A->col_vector();
 
    solver.solve( A.get(), x.get(), b.get(), A_inv.get(), & solve_info );
 
    ct->perm_i2e()->permute( x.get() );
 
    auto                      sol = b->copy();
 
    sol->fill( 1.0 );
    x->axpy( 1.0, sol.get() );
    
    std::cout << "  |x-x~| = " << x->norm2() << std::endl;
 
    DONE();
  }
  catch ( Error & e )
  {
    std::cout << e.to_string() << std::endl;
  }
 
}

PYBIND11_MODULE(integral, m) {
    py::class_<TCoeffFn< real_t >>(m, "TCoeffFnReal");
        // .def("eval", &TCoeffFn< real_t >::eval);

    py::class_<CoeffFnFromPython, TCoeffFn< real_t >>(m, "CoeffFnFromPython")
        .def(py::init<SubmatrixFunction>())
        // .def("eval", &CoeffFnFromPython::eval)
        .def("eval_from_python", &CoeffFnFromPython::eval_from_python);

    m.def("run_problem", &run_problem);
}
