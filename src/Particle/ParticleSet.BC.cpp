//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/**@file ParticleSet.BC.cpp
 * @brief definition of functions controlling Boundary Conditions
 */
#include "Particle/ParticleSetT.h"
#include "Particle/FastParticleOperators.h"
#include "Concurrency/OpenMP.h"
#include "LongRange/StructFact.h"

namespace qmcplusplus
{
/** Creating StructureFactor
 *
 * Currently testing only 1 component for PBCs.
 */
template<typename T>
void ParticleSetT<T>::createSK()
{
  if (structure_factor_)
    throw std::runtime_error("Report bug! structure_factor_ has already been created. Unexpected call sequence.");

  auto& Lattice = getLattice();
  auto& LRBox   = getLRBox();
  if (Lattice.explicitly_defined)
    convert2Cart(R); //make sure that R is in Cartesian coordinates

  if (Lattice.SuperCellEnum != SUPERCELL_OPEN)
  {
    app_log() << "\n  Creating Structure Factor for periodic systems " << LRBox.LR_kc << std::endl;
    structure_factor_ = std::make_unique<StructFact>(LRBox, simulation_cell_.getKLists());
  }

  //set the mass array
  int beforemass = my_species_.numAttributes();
  int massind    = my_species_.addAttribute("mass");
  if (beforemass == massind)
  {
    app_log() << "  ParticleSet::createSK setting mass of  " << getName() << " to 1.0" << std::endl;
    for (int ig = 0; ig < my_species_.getTotalNum(); ++ig)
      my_species_(massind, ig) = 1.0;
  }
  for (int iat = 0; iat < GroupID.size(); iat++)
    Mass[iat] = my_species_(massind, GroupID[iat]);

  coordinates_->setAllParticlePos(R);
}

template<typename T>
void ParticleSetT<T>::turnOnPerParticleSK()
{
  if (structure_factor_)
    structure_factor_->turnOnStorePerParticle(*this);
  else
    throw std::runtime_error("ParticleSet::turnOnPerParticleSK trying to turn on per particle storage in "
                             "structure_factor_ but structure_factor_ has not been created.");
}

template<typename T>
bool ParticleSetT<T>::getPerParticleSKState() const
{
  bool isPerParticleOn = false;
  if (structure_factor_)
    isPerParticleOn = structure_factor_->isStorePerParticle();
  return isPerParticleOn;
}

template<typename T>
void ParticleSetT<T>::convert(const ParticlePos& pin, ParticlePos& pout)
{
  if (pin.getUnit() == pout.getUnit())
  {
    pout = pin;
    return;
  }
  if (pin.getUnit() == PosUnit::Lattice)
  //convert to CartesianUnit
  {
    ConvertPosUnit<ParticlePos, Tensor_t, QMCTraits::DIM>::apply(pin, getLattice().R, pout, 0, pin.size());
  }
  else
  //convert to getLattice()Unit
  {
    ConvertPosUnit<ParticlePos, Tensor_t, QMCTraits::DIM>::apply(pin, getLattice().G, pout, 0, pin.size());
  }
}

template<typename T>
void ParticleSetT<T>::convert2Unit(const ParticlePos& pin, ParticlePos& pout)
{
  pout.setUnit(PosUnit::Lattice);
  if (pin.getUnit() == PosUnit::Lattice)
    pout = pin;
  else
    ConvertPosUnit<ParticlePos, Tensor_t, QMCTraits::DIM>::apply(pin, getLattice().G, pout, 0, pin.size());
}

template<typename T>
void ParticleSetT<T>::convert2Cart(const ParticlePos& pin, ParticlePos& pout)
{
  pout.setUnit(PosUnit::Cartesian);
  if (pin.getUnit() == PosUnit::Cartesian)
    pout = pin;
  else
    ConvertPosUnit<ParticlePos, Tensor_t, QMCTraits::DIM>::apply(pin, getLattice().R, pout, 0, pin.size());
}

template<typename T>
void ParticleSetT<T>::convert2Unit(ParticlePos& pinout)
{
  if (pinout.getUnit() == PosUnit::Lattice)
    return;
  else
  {
    pinout.setUnit(PosUnit::Lattice);
    ConvertPosUnit<ParticlePos, Tensor_t, QMCTraits::DIM>::apply(pinout, getLattice().G, 0, pinout.size());
  }
}

template<typename T>
void ParticleSetT<T>::convert2Cart(ParticlePos& pinout)
{
  if (pinout.getUnit() == PosUnit::Cartesian)
    return;
  else
  {
    pinout.setUnit(PosUnit::Cartesian);
    ConvertPosUnit<ParticlePos, Tensor_t, QMCTraits::DIM>::apply(pinout, getLattice().R, 0, pinout.size());
  }
}

template<typename T>
void ParticleSetT<T>::applyBC(const ParticlePos& pin, ParticlePos& pout)
{
  applyBC(pin, pout, 0, pin.size());
}

template<typename T>
void ParticleSetT<T>::applyBC(const ParticlePos& pin, ParticlePos& pout, int first, int last)
{
  if (pin.getUnit() == PosUnit::Cartesian)
  {
    if (pout.getUnit() == PosUnit::Cartesian)
      ApplyBConds<ParticlePos, Tensor_t, QMCTraits::DIM>::Cart2Cart(pin, getLattice().G, getLattice().R, pout, first,
                                                                    last);
    else if (pout.getUnit() == PosUnit::Lattice)
      ApplyBConds<ParticlePos, Tensor_t, QMCTraits::DIM>::Cart2Unit(pin, getLattice().G, pout, first, last);
    else
      throw std::runtime_error("Unknown unit conversion");
  }
  else if (pin.getUnit() == PosUnit::Lattice)
  {
    if (pout.getUnit() == PosUnit::Cartesian)
      ApplyBConds<ParticlePos, Tensor_t, QMCTraits::DIM>::Unit2Cart(pin, getLattice().R, pout, first, last);
    else if (pout.getUnit() == PosUnit::Lattice)
      ApplyBConds<ParticlePos, Tensor_t, QMCTraits::DIM>::Unit2Unit(pin, pout, first, last);
    else
      throw std::runtime_error("Unknown unit conversion");
  }
  else
    throw std::runtime_error("Unknown unit conversion");
}

template<typename T>
void ParticleSetT<T>::applyBC(ParticlePos& pos)
{
  if (pos.getUnit() == PosUnit::Lattice)
  {
    ApplyBConds<ParticlePos, Tensor_t, QMCTraits::DIM>::Unit2Unit(pos, 0, TotalNum);
  }
  else
  {
    ApplyBConds<ParticlePos, Tensor_t, QMCTraits::DIM>::Cart2Cart(pos, getLattice().G, getLattice().R, 0, TotalNum);
  }
}

template<typename T>
void ParticleSetT<T>::applyMinimumImage(ParticlePos& pinout)
{
  if (getLattice().SuperCellEnum == SUPERCELL_OPEN)
    return;
  for (int i = 0; i < pinout.size(); ++i)
    getLattice().applyMinimumImage(pinout[i]);
}

template<typename T>
void ParticleSetT<T>::convert2UnitInBox(const ParticlePos& pin, ParticlePos& pout)
{
  pout.setUnit(PosUnit::Lattice);
  convert2Unit(pin, pout); // convert to crystalline unit
  put2box(pout);
}

template<typename T>
void ParticleSetT<T>::convert2CartInBox(const ParticlePos& pin, ParticlePos& pout)
{
  convert2UnitInBox(pin, pout); // convert to crystalline unit
  convert2Cart(pout);
}

template class ParticleSetT<double>;
template class ParticleSetT<float>;
template class ParticleSetT<std::complex<double>>;
template class ParticleSetT<std::complex<float>>;

} // namespace qmcplusplus
