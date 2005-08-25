//////////////////////////////////////////////////////////////////
// (c) Copyright 1998-2002 by Jeongnim Kim
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//   Jeongnim Kim
//   National Center for Supercomputing Applications &
//   Materials Computation Center
//   University of Illinois, Urbana-Champaign
//   Urbana, IL 61801
//   e-mail: jnkim@ncsa.uiuc.edu
//   Tel:    217-244-6319 (NCSA) 217-333-3324 (MCC)
//
// Supported by 
//   National Center for Supercomputing Applications, UIUC
//   Materials Computation Center, UIUC
//   Department of Physics, Ohio State University
//   Ohio Supercomputer Center
//////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef OHMMSQMC_DIRACDETERMINANT_H
#define OHMMSQMC_DIRACDETERMINANT_H
#include "Numerics/DeterminantOperators.h"
#include "Numerics/OhmmsBlas.h"

namespace ohmmsqmc {

  class ParticleSet;
  class WalkerSetRef;

  /** Generic class to handle a DiracDeterminant.
   *
   *The (S)ingle(P)article(O)rbitalSet template parameter is an 
   *engine which fills in single-particle orbital terms.
   *
   *The DiracDeterminant<SPOSet> class handles the determinant and co-factors:
   *<ul>
   *<li> set up the matrix \f$D\f$ from \f$ \{\psi\} \f$ (the set of 
   *single-particle orbitals) such that \f$D_{ij} =  \psi_j({\bf r}_i).\f$
   *<li> invert the matrix \f$D.\f$
   *<li> evaluate the determinant \f$|D|\f$ and the gradient and 
   *laplacian of the logarithm of the determinant.
   *</ul>
   *The template class SPOSet has to evaluate
   *<ul>
   *<li> psiM(j,i) \f$= \psi_j({\bf r}_i)\f$
   *<li> dpsiM(i,j) \f$= \nabla_i \psi_j({\bf r}_i)\f$
   *<li> d2psiM(i,j) \f$= \nabla_i^2 \psi_j({\bf r}_i)\f$
   *</ul>
   *Very important to note that psiM is a transpose of a matrix defined as
   *\f$D_{ij} =  \psi_j({\bf r}_i)\f$. Upon inversion, other operations
   *to update and evaluate \f$ \nabla_i \ln{\rm D} \f$ and 
   *\f$ \nabla_i^2\ln{\rm D} \f$ can be done efficiently by dot products.
   *
   *Since each determinant operates on a subset of particles (e.g., up
   *or down electrons), auxiliary indices (first, last)
   *are passed to enable operations only on the particles belonging to 
   *this DiracDeterminant term.
   *
   *In evaluating the determinant it is necessary to define the
   *following quatities.  
   *<ul>
   *<li> The minor \f$ M_{ij} \f$ for the element \f$ D_{ij} \f$ 
   *is the determinant of an \f$ (N-1) \times (N-1) \f$ matrix obtained 
   *by removing all the elements of row \f$ i \f$ and column \f$ j \f$.  
   *<li>The cofactor matrix \f$ C \f$ is constructed on an element by 
   *element basis from the minor using the simple relation: 
   *\f[ C_{ij} = (-1)^{i+j}M_{ij} \f]
   *<li>The inverse of a matrix is related to the transpose of the 
   *cofactor matrix by the relation
   \f[ (D^{-1})_{ij} = \frac{(C_{ij})^T}{|D|} = \frac{C_{ji}}{|D|} \f]
   *</ul>
   *Using these definitions the determinant is evaluated as the sum
   *of the products of the elements of any row or
   *column of the matrix and the corresponding cofactor
   \f[
   |D| = \sum_{j=1}^N D_{ij}C_{ij}\;\;\longrightarrow \mbox{along row i}
   \f]
   *
   *To calculate the local energy \f$E_L\f$ it is necessary to evaluate
   *the gradient and laplacian of the logarithm of the Diracdeterminant
   \f[ 
   \nabla_i |D({\bf R})| = \frac{\nabla_i|D({\bf R})|}{|D({\bf R})|} 
   \f]
   \f[
   \nabla^2_i |D({\bf R})| = \frac{\nabla_i^2|D({\bf R})|}{|D({\bf R})|}
   -\left(\frac{\nabla_i|D({\bf R})|}{|D({\bf R})|}\right)^2.
   \f]
   *We have already shown how to evaluate the determinant in terms of its
   *cofactors, taking the derivatives follows from this  
   \f[
   \nabla_i|D| = |D|\sum_{j=1}^N (\nabla_i D_{ij})(D^{-1})_{ji},
   \f]
   *where \f$ D_{ij} \f$ is a function of \f${\bf r_i}\f$ and 
   *\f$ (D^{-1})_{ji} \f$ is a function of all of the other coordinates 
   *except \f${\bf r_i}\f$.  This leads to the result
   \f[
   \frac{\nabla_i|D|}{|D|} = \sum_{j=1}^N (\nabla_i D_{ij})(D^{-1})_{ji}.
   \f]
   \f[
   \frac{\nabla^2_i|D|}{|D|} = \sum_{j=1}^N (\nabla^2_i D_{ij})(D^{-1})_{ji}.
   \f]
   *
   *@note SlaterDeterminant is a product of DiracDeterminants.

   */
  template<class SPOSet>
  struct DiracDeterminant: public OrbitalBase {

#if defined(USE_BLITZ)
    typedef blitz::Array<ValueType,2> Determinant_t;
    typedef blitz::Array<GradType,2>  Gradient_t;
    typedef blitz::Array<ValueType,2> Laplacian_t;
#else
    typedef Matrix<ValueType> Determinant_t;
    typedef Matrix<GradType>  Gradient_t;
    typedef Matrix<ValueType> Laplacian_t;
#endif

    /** constructor
     *@param spos the single-particle orbital set
     *@param first index of the first particle
     */
    DiracDeterminant(SPOSet& spos, int first=0): 
      NP(0), Phi(spos), FirstIndex(first) {}

    ///default destructor
    ~DiracDeterminant() {}
  
    /**copy constructor
     *@brief copy constructor, only resize and assign orbitals
     */
    DiracDeterminant(const DiracDeterminant<SPOSet>& s): Phi(s.Phi),NP(0){
      resize(s.rows(), s.cols());
    }

    DiracDeterminant<SPOSet>& operator=(const DiracDeterminant<SPOSet>& s) {
      NP=0;
      resize(s.rows(), s.cols());
      return *this;
    }

    /** set the index of the first particle in the determinant and reset the size of the determinant
     *@param first index of first particle
     *@param nel number of particles in the determinant
     */
    inline void set(int first, int nel) {
      FirstIndex = first;
      resize(nel,nel);
    }

    ///reset the single-particle orbital set
    inline void reset() { Phi.reset(); }
   
    ///reset the size: with the number of particles and number of orbtials
    inline void resize(int nel, int morb) {
      int norb=morb;
      if(norb <= 0) norb = nel; // for morb == -1 (default)
      psiM.resize(nel,norb);
      dpsiM.resize(nel,norb);
      d2psiM.resize(nel,norb);
      psiM_temp.resize(nel,norb);
      dpsiM_temp.resize(nel,norb);
      d2psiM_temp.resize(nel,norb);
      psiMinv.resize(nel,norb);
      psiV.resize(norb);
      LastIndex = FirstIndex + nel;
      NumPtcls=nel;
      NumOrbitals=norb;
    }

    ValueType registerData(ParticleSet& P, PooledData<RealType>& buf) {

      if(NP == 0) {//first time, allocate once
	//int norb = cols();
	dpsiV.resize(NumOrbitals);
	d2psiV.resize(NumOrbitals);
	workV1.resize(NumOrbitals);
	workV2.resize(NumOrbitals);
	NP=P.getTotalNum();
	myG.resize(NP);
	myL.resize(NP);
	myG_temp.resize(NP);
	myL_temp.resize(NP);
	FirstAddressOfG = &myG[0][0];
	LastAddressOfG = FirstAddressOfG + NP*DIM;
	FirstAddressOfdV = &(dpsiM(0,0)[0]); //(*dpsiM.begin())[0]);
	LastAddressOfdV = FirstAddressOfdV + NumPtcls*NumOrbitals*DIM;
      }

      //allocate once but each walker calls this
      myG=0.0;
      myL=0.0;

      ValueType x=evaluate(P,myG,myL); 

      P.G += myG;
      P.L += myL;

      //add the data: determinant, inverse, gradient and laplacians
      buf.add(psiM.begin(),psiM.end());
      buf.add(FirstAddressOfdV,LastAddressOfdV);
      buf.add(d2psiM.begin(),d2psiM.end());
      buf.add(myL.begin(), myL.end());
      buf.add(FirstAddressOfG,LastAddressOfG);
      buf.add(CurrentDet);

      return CurrentDet;
    }

    ValueType updateBuffer(ParticleSet& P, PooledData<RealType>& buf) {

      myG=0.0;
      myL=0.0;
      ValueType x=evaluate(P,myG,myL); 
      P.G += myG;
      P.L += myL;
      buf.put(psiM.begin(),psiM.end());
      buf.put(FirstAddressOfdV,LastAddressOfdV);
      buf.put(d2psiM.begin(),d2psiM.end());
      buf.put(myL.begin(), myL.end());
      buf.put(FirstAddressOfG,LastAddressOfG);
      buf.put(CurrentDet);

      return CurrentDet;
    }

    void copyFromBuffer(ParticleSet& P, PooledData<RealType>& buf) {

      buf.get(psiM.begin(),psiM.end());
      buf.get(FirstAddressOfdV,LastAddressOfdV);
      buf.get(d2psiM.begin(),d2psiM.end());
      buf.get(myL.begin(), myL.end());
      buf.get(FirstAddressOfG,LastAddressOfG);
      buf.get(CurrentDet);

      //re-evaluate it for testing
      //Phi.evaluate(P, FirstIndex, LastIndex, psiM, dpsiM, d2psiM);
      //CurrentDet = Invert(psiM.data(),NumPtcls,NumOrbitals);

      //need extra copy for gradient/laplacian calculations without updating it
      psiM_temp = psiM;
      dpsiM_temp = dpsiM;
      d2psiM_temp = d2psiM;
    }

    /** dump the inverse to the buffer
     */
    inline void dumpToBuffer(ParticleSet& P, PooledData<RealType>& buf) {
      buf.add(psiM.begin(),psiM.end());
    }

    /** copy the inverse from the buffer
     */
    inline void dumpFromBuffer(ParticleSet& P, PooledData<RealType>& buf) {
      buf.get(psiM.begin(),psiM.end());
    }

    /** return the ratio only for the  iat-th partcle move
     * @param P current configuration
     * @param iat the particle thas is being moved
     */
    inline ValueType ratio(ParticleSet& P, int iat) {
      Phi.evaluate(P, iat, psiV);
      return DetRatio(psiM, psiV.begin(),iat-FirstIndex);
      //return curRatio= DetRatio(psiM, psiV.begin(),iat-FirstIndex);
      //return curRatio = BLAS::dot(NumOrbitals,psiM[iat-FirstIndex],&psiV[0]);
      //return curRatio= detRatio(psiM_temp[iat-FirstIndex], psiV.begin(),NumOrbitals);
    }

    /** return the ratio
     * @param P current configuration
     * @param iat particle whose position is moved
     * @param dG differential Gradients
     * @param dL differential Laplacians
     *
     * Data member *_temp contain the data assuming that the move is accepted
     * and are used to evaluate differential Gradients and Laplacians.
     */
    ValueType ratio(ParticleSet& P, int iat,
		    ParticleSet::ParticleGradient_t& dG, 
		    ParticleSet::ParticleLaplacian_t& dL) {

      Phi.evaluate(P, iat, psiV, dpsiV, d2psiV);
      WorkingIndex = iat-FirstIndex;

      curRatio= DetRatio(psiM_temp, psiV.begin(),WorkingIndex);
      //curRatio = BLAS::dot(NumOrbitals,psiM_temp[WorkingIndex],&psiV[0]);
      //curRatio= detRatio(psiM_temp[WorkingIndex], psiV.begin(), NumOrbitals);

      //update psiM_temp with the row substituted
      DetUpdate(psiM_temp,psiV,workV1,workV2,WorkingIndex,curRatio);

      //update dpsiM_temp and d2psiM_temp 
      for(int j=0; j<NumOrbitals; j++) {
	dpsiM_temp(WorkingIndex,j)=dpsiV[j];
	d2psiM_temp(WorkingIndex,j)=d2psiV[j];
      }

      int kat=FirstIndex;
      for(int i=0; i<NumPtcls; i++,kat++) {
	PosType rv =psiM_temp(i,0)*dpsiM_temp(i,0);
	ValueType lap=psiM_temp(i,0)*d2psiM_temp(i,0);
        for(int j=1; j<NumOrbitals; j++) {
	  rv += psiM_temp(i,j)*dpsiM_temp(i,j);
	  lap += psiM_temp(i,j)*d2psiM_temp(i,j);
	}
	lap -= dot(rv,rv);
	dG[kat] += rv - myG[kat];  myG_temp[kat]=rv;
	dL[kat] += lap -myL[kat];  myL_temp[kat]=lap;
      }

      return curRatio;
    }

    ValueType logRatio(ParticleSet& P, int iat,
		    ParticleSet::ParticleGradient_t& dG, 
		    ParticleSet::ParticleLaplacian_t& dL) {
      ValueType r=ratio(P,iat,dG,dL);
      SignValue = (r<0.0)? -1.0: 1.0;
      return log(abs(r));
    }


    /** move was accepted, update the real container
     */
    void update(ParticleSet& P, int iat) {
      CurrentDet *= curRatio;
      myG = myG_temp;
      myL = myL_temp;
      psiM = psiM_temp;
      for(int j=0; j<NumOrbitals; j++) {
	dpsiM(WorkingIndex,j)=dpsiV[j];
	d2psiM(WorkingIndex,j)=d2psiV[j];
      }
      curRatio=1.0;
    }

    /** move was rejected. copy the real container to the temporary to move on
     */
    void restore(int iat) {
      psiM_temp = psiM;
      for(int j=0; j<NumOrbitals; j++) {
	dpsiM_temp(WorkingIndex,j)=dpsiM(WorkingIndex,j);
	d2psiM_temp(WorkingIndex,j)=d2psiM(WorkingIndex,j);
      }
      curRatio=1.0;
    }

    
    void update(ParticleSet& P, 
		ParticleSet::ParticleGradient_t& dG, 
		ParticleSet::ParticleLaplacian_t& dL,
		int iat) {

      DetUpdate(psiM,psiV,workV1,workV2,WorkingIndex,curRatio);
      for(int j=0; j<NumOrbitals; j++) {
	dpsiM(WorkingIndex,j)=dpsiV[j];
	d2psiM(WorkingIndex,j)=d2psiV[j];
      }

      int kat=FirstIndex;
      for(int i=0; i<NumPtcls; i++,kat++) {
	PosType rv =psiM(i,0)*dpsiM(i,0);
	ValueType lap=psiM(i,0)*d2psiM(i,0);
	for(int j=1; j<NumOrbitals; j++) {
	  rv += psiM(i,j)*dpsiM(i,j);
	  lap += psiM(i,j)*d2psiM(i,j);
	}
	lap -= dot(rv,rv);
	dG[kat] += rv - myG[kat]; myG[kat]=rv;
	dL[kat] += lap -myL[kat]; myL[kat]=lap;
      }

      //not very useful
      CurrentDet *= curRatio;
      curRatio=1.0;
    }

    ValueType evaluate(ParticleSet& P, PooledData<RealType>& buf) {

      buf.put(psiM.begin(),psiM.end());
      buf.put(FirstAddressOfdV,LastAddressOfdV);
      buf.put(d2psiM.begin(),d2psiM.end());
      buf.put(myL.begin(), myL.end());
      buf.put(FirstAddressOfG,LastAddressOfG);
      buf.put(CurrentDet);

      return CurrentDet;
    }

    void resizeByWalkers(int nwalkers);

    ///return the number of rows (or the number of electrons)
    inline int rows() const { return NumPtcls;}
    //inline int rows() const { return psiM.rows();}

    ///return the number of coloumns  (or the number of orbitals)
    inline int cols() const { return NumOrbitals;}
    //inline int cols() const { return psiM.cols();}

    ///evaluate log of determinant for a particle set: should not be called 
    ValueType
    evaluateLog(ParticleSet& P, 
	        ParticleSet::ParticleGradient_t& G, 
	        ParticleSet::ParticleLaplacian_t& L) {
      std::cerr << "DiracDeterminant::evaluateLog should never be called directly" << std::endl;
      return 0.0;
    }

    ///evaluate for a particle set
    ValueType
    evaluate(ParticleSet& P, 
	     ParticleSet::ParticleGradient_t& G, 
	     ParticleSet::ParticleLaplacian_t& L);

    ///evaluate for walkers
    void 
    evaluate(WalkerSetRef& W, 
	     ValueVectorType& psi,
	     WalkerSetRef::WalkerGradient_t& G,
	     WalkerSetRef::WalkerLaplacian_t& L);

    ///The number of particles
    int NP;

    int NumOrbitals;
    int NumPtcls;

    ///index of the first particle with respect to the particle set
    int FirstIndex;

    ///index of the last particle with respect to the particle set
    int LastIndex;

    ///a set of single-particle orbitals used to fill in the  values of the matrix 
    SPOSet& Phi;

    ///index of the particle (or row) 
    int WorkingIndex;      

    ///Current determinant value
    ValueType CurrentDet;

    /// psiM(j,i) \f$= \psi_j({\bf r}_i)\f$
    Determinant_t psiM, psiM_temp;

    /// temporary container for testing
    Determinant_t psiMinv;

    /// dpsiM(i,j) \f$= \nabla_i \psi_j({\bf r}_i)\f$
    Gradient_t    dpsiM, dpsiM_temp;

    /// d2psiM(i,j) \f$= \nabla_i^2 \psi_j({\bf r}_i)\f$
    Laplacian_t   d2psiM, d2psiM_temp;

    /// value of single-particle orbital for particle-by-particle update
    std::vector<ValueType> psiV;
    std::vector<GradType> dpsiV;
    std::vector<ValueType> d2psiV;
    std::vector<ValueType> workV1, workV2;

    ///storages to process many walkers once
    vector<Determinant_t> psiM_v; 
    vector<Gradient_t>    dpsiM_v; 
    vector<Laplacian_t>   d2psiM_v; 

    ValueType curRatio,cumRatio;
    ValueType *FirstAddressOfG;
    ValueType *LastAddressOfG;
    ValueType *FirstAddressOfdV;
    ValueType *LastAddressOfdV;

    ParticleSet::ParticleGradient_t myG, myG_temp;
    ParticleSet::ParticleLaplacian_t myL, myL_temp;
  };

  /** Calculate the value of the Dirac determinant for particles
   *@param P input configuration containing N particles
   *@param G a vector containing N gradients
   *@param L a vector containing N laplacians
   *@return the value of the determinant
   *
   *\f$ (first,first+nel). \f$  Add the gradient and laplacian 
   *contribution of the determinant to G(radient) and L(aplacian)
   *for local energy calculations.
   */
  template<class SPOSet>
  inline 
  typename DiracDeterminant<SPOSet>::ValueType 
  DiracDeterminant<SPOSet>::evaluate(ParticleSet& P, 
				     ParticleSet::ParticleGradient_t& G, 
				     ParticleSet::ParticleLaplacian_t& L) {

    Phi.evaluate(P, FirstIndex, LastIndex, psiM,dpsiM, d2psiM);
    CurrentDet = Invert(psiM.data(),NumPtcls,NumOrbitals);
    int iat = FirstIndex; //the index of the particle with respect to P
    for(int i=0; i<NumPtcls; i++, iat++) {
      PosType rv = psiM(i,0)*dpsiM(i,0);
      ValueType lap=psiM(i,0)*d2psiM(i,0);
      for(int j=1; j<NumOrbitals; j++) {
	rv += psiM(i,j)*dpsiM(i,j);
	lap += psiM(i,j)*d2psiM(i,j);
      }
      G(iat) += rv;
      L(iat) += lap - dot(rv,rv);
    }
    return CurrentDet;
  }


  /**void evaluate(WalkerSetRef& W,  WfsVec& psi, GradMat& G, LapMat& L)
   *@param W Walkers, set of input configurations, Nw is the number of walkers
   *@param psi a vector containing Nw determinants
   *@param G a matrix containing Nw x N gradients
   *@param L a matrix containing Nw x N laplacians
   *@brief N is the number of particles per walker and Nw is the number of walkers.
   *Designed for vectorized move, i.e., all the walkers move simulatenously as
   *in molecu. While calculating the determinant values for a set of walkers,
   *add the gradient and laplacian contribution of the determinant
   *to G and L for local energy calculations.
   */
  template<class SPOSet>
  inline void 
  DiracDeterminant<SPOSet>::evaluate(WalkerSetRef& W, 
				     ValueVectorType& psi,
				     WalkerSetRef::WalkerGradient_t& G,
				     WalkerSetRef::WalkerLaplacian_t& L) {

    int nw = W.walkers();

    //evaluate \f$(D_{ij})^t\f$ and other quantities for gradient/laplacians
    Phi.evaluate(W, FirstIndex, LastIndex, psiM_v, dpsiM_v, d2psiM_v);
    //int nrows = rows();
    //int ncols = cols();
    
    for(int iw=0; iw< nw; iw++) {
      psi[iw] *= Invert(psiM_v[iw].data(),NumPtcls,NumOrbitals);
      int iat = FirstIndex; //the index of the particle with respect to P
      const Determinant_t& logdet = psiM_v[iw];
      const Gradient_t& dlogdet = dpsiM_v[iw];
      const Laplacian_t& d2logdet = d2psiM_v[iw];

      for(int i=0; i<NumPtcls; i++, iat++) {
	PosType rv = logdet(i,0)*dlogdet(i,0);
	ValueType lap=logdet(i,0)*d2logdet(i,0);
	for(int j=1; j<NumOrbitals; j++) {
	  rv += logdet(i,j)*dlogdet(i,j);
	  lap += logdet(i,j)*d2logdet(i,j);
	}
	G(iw,iat) += rv;
	L(iw,iat) += lap - dot(rv,rv);
      }
    }
  }
  
  template<class SPOSet>
  void DiracDeterminant<SPOSet>::resizeByWalkers(int nwalkers) {
    if(psiM_v.size() < nwalkers) {
      psiM_v.resize(nwalkers);
      dpsiM_v.resize(nwalkers);
      d2psiM_v.resize(nwalkers);
      for(int iw=0; iw<nwalkers; iw++) psiM_v[iw].resize(NumPtcls, NumOrbitals);
      for(int iw=0; iw<nwalkers; iw++) dpsiM_v[iw].resize(NumPtcls, NumOrbitals);
      for(int iw=0; iw<nwalkers; iw++) d2psiM_v[iw].resize(NumPtcls, NumOrbitals);
    }
    Phi.resizeByWalkers(nwalkers);
  }
}
#endif
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$ 
 ***************************************************************************/
