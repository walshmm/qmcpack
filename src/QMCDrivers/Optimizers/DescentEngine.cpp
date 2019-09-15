
//Prototype code for a descent engine
//Some functions modeled on those in linear method engine, but trying to avoid use of formic wrappers for things like mpi.
//Long-term(?) amibition is to eventually get rid of most of formic except for parts essential for the adaptive three shift LM and BLM.


#include <cmath>
#include <vector>
#include <string>
#include "QMCDrivers/Optimizers/DescentEngine.h"
#include "OhmmsData/ParameterSet.h"
#include "Message/CommOperators.h"

namespace qmcplusplus
{
DescentEngine::DescentEngine(Communicate* comm, const xmlNodePtr cur)
    : myComm(comm),
      engineTargetExcited(false),
      numParams(0),
      flavor("RMSprop"),
      TJF_2Body_eta(.01),
      TJF_1Body_eta(.01),
      F_eta(.001),
      Gauss_eta(.001),
      CI_eta(.01),
      Orb_eta(.001),
      ramp_eta(false),
      ramp_num(30),
      store_num(5)
{
  descent_num_ = 0;
  store_count  = 0;
  processXML(cur);
}


bool DescentEngine::processXML(const xmlNodePtr cur)
{
  std::string excited("no");
  std::string ramp_etaStr("no");

  ParameterSet m_param;
  m_param.add(excited, "targetExcited", "string");
  //Type of descent method being used
  m_param.add(flavor, "flavor", "string");
  m_param.add(TJF_2Body_eta, "TJF_2Body_eta", "double");
  m_param.add(TJF_1Body_eta, "TJF_1Body_eta", "double");
  m_param.add(F_eta, "F_eta", "double");
  m_param.add(CI_eta, "CI_eta", "double");
  m_param.add(Gauss_eta, "Gauss_eta", "double");
  m_param.add(Orb_eta, "Orb_eta", "double");
  m_param.add(ramp_etaStr, "Ramp_eta", "string");
  m_param.add(ramp_num, "Ramp_num", "int");
  m_param.add(store_num, "Stored_Vectors", "int");
  m_param.put(cur);

  engineTargetExcited = (excited == "yes");

  ramp_eta = (ramp_etaStr == "yes");


  return true;
}

void DescentEngine::prepareStorage(const int num_replicas, const int num_optimizables)
{
  avg_le_der_samp_.resize(num_optimizables);
  avg_der_rat_samp_.resize(num_optimizables);
  LDerivs.resize(num_optimizables);

  numParams = num_optimizables;

  std::fill(avg_le_der_samp_.begin(), avg_le_der_samp_.end(), 0.0);
  std::fill(avg_der_rat_samp_.begin(), avg_der_rat_samp_.end(), 0.0);

  replica_le_der_samp_.resize(num_replicas);
  replica_der_rat_samp_.resize(num_replicas);

  for (int i = 0; i < num_replicas; i++)
  {
    replica_le_der_samp_[i].resize(num_optimizables);
    std::fill(replica_le_der_samp_[i].begin(), replica_le_der_samp_[i].end(), 0.0);

    replica_der_rat_samp_[i].resize(num_optimizables);
    std::fill(replica_der_rat_samp_[i].begin(), replica_der_rat_samp_[i].end(), 0.0);
  }

  w_sum       = 0;
  e_avg       = 0;
  e_sum       = 0;
  eSquare_sum = 0;
  eSquare_avg = 0;
}

void DescentEngine::setEtemp(const std::vector<double>& etemp)
{
  e_sum       = etemp[0];
  w_sum       = etemp[1];
  eSquare_sum = etemp[2];
  e_avg       = e_sum / w_sum;
  eSquare_avg = eSquare_sum / w_sum;

  app_log() << "e_sum: " << e_sum << std::endl;
  app_log() << "w_sum: " << w_sum << std::endl;
  app_log() << "e_avg: " << e_avg << std::endl;
  app_log() << "eSquare_sum: " << eSquare_sum << std::endl;
  app_log() << "eSquare_avg: " << eSquare_avg << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief  Function that Take Sample Data from the Host Code
///
/// \param[in]  der_rat_samp   <n|Psi_i>/<n|Psi> (i = 0 (|Psi>), 1, ... N_var )
/// \param[in]  le_der_samp    <n|H|Psi_i>/<n|Psi> (i = 0 (|Psi>), 1, ... N_var )
/// \param[in]  ls_der_samp    <|S^2|Psi_i>/<n|Psi> (i = 0 (|Psi>), 1, ... N_var )
/// \param[in]  vgs_samp       |<n|value_fn>/<n|guiding_fn>|^2
/// \param[in]  weight_samp    weight for this sample
///
////////////////////////////////////////////////////////////////////////////////////////////////////////
void DescentEngine::takeSample(const int replica_id,
                               const std::vector<double>& der_rat_samp,
                               const std::vector<double>& le_der_samp,
                               const std::vector<double>& ls_der_samp,
                               double vgs_samp,
                               double weight_samp)
{
  const size_t num_optimizables = der_rat_samp.size() - 1;

  for (int i = 0; i < num_optimizables; i++)
  {
    replica_le_der_samp_[replica_id].at(i) += le_der_samp.at(i + 1);
    replica_der_rat_samp_[replica_id].at(i) += der_rat_samp.at(i + 1);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief  Function that Take Sample Data from the Host Code
///
/// \param[in]  local_en       local energy
/// \param[in]  vgs_samp       |<n|value_fn>/<n|guiding_fn>|^2
/// \param[in]  weight_samp    weight for this sample
///
////////////////////////////////////////////////////////////////////////////////////////////////////////
void DescentEngine::takeSample(double local_en, double vgs_samp, double weight_samp) {}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief  Function that reduces all vector information from all processors to the root
///         processor
///
////////////////////////////////////////////////////////////////////////////////////////////////////////
void DescentEngine::sample_finish()
{
  for (int i = 0; i < replica_le_der_samp_.size(); i++)
  {
    for (int j = 0; j < LDerivs.size(); j++)
    {
      avg_le_der_samp_[j] += replica_le_der_samp_[i].at(j);
      avg_der_rat_samp_[j] += replica_der_rat_samp_[i].at(j);
    }
  }


  myComm->allreduce(avg_le_der_samp_);
  myComm->allreduce(avg_der_rat_samp_);

  for (int i = 0; i < LDerivs.size(); i++)
  {
    avg_le_der_samp_.at(i)  = avg_le_der_samp_.at(i) / w_sum;
    avg_der_rat_samp_.at(i) = avg_der_rat_samp_.at(i) / w_sum;

    app_log() << "Parameter # " << i << " Hamiltonian term: " << avg_le_der_samp_.at(i) << std::endl;
    app_log() << "Parameter # " << i << " Overlap term: " << avg_der_rat_samp_.at(i) << std::endl;

    //Computation of averaged derivatives for excited state functional will be added in future
    if (!engineTargetExcited)
    {
      LDerivs.at(i) = 2 * avg_le_der_samp_.at(i) - e_avg * (2 * avg_der_rat_samp_.at(i));
    }
  }
}


//Function for updating parameters during descent optimization
void DescentEngine::updateParameters()
{
  app_log() << "Number of Parameters: " << numParams << std::endl;

  app_log() << "Parameter Type step sizes: "
            << " TJF_2Body_eta=" << TJF_2Body_eta << " TJF_1Body_eta=" << TJF_1Body_eta << " F_eta=" << F_eta
            << " CI_eta=" << CI_eta << " Orb_eta=" << Orb_eta << std::endl;

  // Get set of derivatives for current (kth) optimization step
  std::vector<double> curDerivSet = derivRecords.at(derivRecords.size() - 1);
  std::vector<double> prevDerivSet;

  if (!taus.empty())
  {
    // Get set of derivatives for previous (k-1th) optimization step
    prevDerivSet = derivRecords.at(derivRecords.size() - 2);
  }

  double denom;
  double numer;
  double v;
  double corNumer;
  double corV;

  double epsilon = 1e-8;
  double type_Eta;

  double tau;
  // Update parameters according to specified flavor of gradient descent method

  // RMSprop corresponds to the method used by Booth and co-workers
  if (flavor.compare("RMSprop") == 0)
  {
    app_log() << "Using RMSprop" << std::endl;

    // To match up with Booth group paper notation, prevLambda is lambda_k-1,
    // curLambda is lambda_k, nextLambda is lambda_k+1
    double curLambda  = .5 + .5 * std::sqrt(1 + 4 * lambda * lambda);
    double nextLambda = .5 + .5 * std::sqrt(1 + 4 * curLambda * curLambda);
    double gamma      = (1 - curLambda) / nextLambda;

    // Define damping factor that turns off acceleration of the algorithm
    // small value of d corresponds to quick damping and effectively using
    // steepest descent
    double d           = 100;
    double decayFactor = std::exp(-(1 / d) * (descent_num_));
    gamma              = gamma * decayFactor;

    double rho = .9;

    for (int i = 0; i < numParams; i++)
    {
      double curSquare = std::pow(curDerivSet.at(i), 2);

      // Need to calculate step size tau for each parameter inside loop
      // In RMSprop, the denominator of the step size depends on a a running average of past squares of the parameter derivative
      if (derivsSquared.size() < numParams)
      {
        curSquare = std::pow(curDerivSet.at(i), 2);
      }
      else if (derivsSquared.size() >= numParams)
      {
        curSquare = rho * derivsSquared.at(i) + (1 - rho) * std::pow(curDerivSet.at(i), 2);
      }

      denom = std::sqrt(curSquare + epsilon);

      //The numerator of the step size is set according to parameter type based on input choices
      type_Eta = this->setStepSize(i);
      tau      = type_Eta / denom;

      // Include an additional factor to cause step size to eventually decrease to 0 as number of steps taken increases
      double stepLambda = .1;

      double stepDecayDenom = 1 + stepLambda * descent_num_;
      tau                   = tau / stepDecayDenom;


      //Update parameter values
      //If case corresponds to being after the first descent step
      if (taus.size() >= numParams)
      {
        double oldTau = taus.at(i);

        currentParams.at(i) = (1 - gamma) * (currentParams.at(i) - tau * curDerivSet.at(i)) +
            gamma * (paramsCopy.at(i) - oldTau * prevDerivSet.at(i));
      }
      else
      {
        tau = type_Eta;

        currentParams.at(i) = currentParams.at(i) - tau * curDerivSet.at(i);
      }

      if (taus.size() < numParams)
      {
        // For the first optimization step, need to add to the vectors
        taus.push_back(tau);
        derivsSquared.push_back(curSquare);
      }
      else
      {
        // When not on the first step, can overwrite the previous stored values
        taus[i]          = tau;
        derivsSquared[i] = curSquare;
      }

      paramsCopy[i] = currentParams[i];
    }

    // Store current (kth) lambda value for next optimization step
    lambda = curLambda;
  }
  // Random uses only the sign of the parameter derivatives and takes a step of random size within a range.
  else if (flavor.compare("Random") == 0)
  {
    app_log() << "Using Random" << std::endl;

    for (int i = 0; i < numParams; i++)
    {
      denom        = 1;
      double alpha = ((double)rand() / RAND_MAX);
      double sign  = std::abs(curDerivSet[i]) / curDerivSet[i];
      if (std::isnan(sign))
      {
        app_log() << "Got a nan, choosing sign randomly with 50-50 probability" << std::endl;

        double t = ((double)rand() / RAND_MAX);
        if (t > .5)
        {
          sign = 1;
        }
        else
        {
          sign = -1;
        }
      }
      app_log() << "This is random alpha: " << alpha << " with sign: " << sign << std::endl;

      currentParams.at(i) = currentParams.at(i) - tau * alpha * sign;
    }
  }

  else
  {
    // ADAM method
    if (flavor.compare("ADAM") == 0)
    {
      app_log() << "Using ADAM" << std::endl;

      for (int i = 0; i < numParams; i++)
      {
        double curSquare = std::pow(curDerivSet.at(i), 2);
        double beta1     = .9;
        double beta2     = .99;
        if (descent_num_ == 0)
        {
          numerRecords.push_back(0);
          denomRecords.push_back(0);
        }
        numer = beta1 * numerRecords[i] + (1 - beta1) * curDerivSet[i];
        v     = beta2 * denomRecords[i] + (1 - beta2) * curSquare;

        corNumer = numer / (1 - std::pow(beta1, descent_num_ + 1));
        corV     = v / (1 - std::pow(beta2, descent_num_ + 1));

        denom = std::sqrt(corV) + epsilon;

        type_Eta = this->setStepSize(i);
        tau      = type_Eta / denom;

        currentParams.at(i) = currentParams.at(i) - tau * corNumer;

        if (taus.size() < numParams)
        {
          // For the first optimization step, need to add to the vectors
          taus.push_back(tau);
          derivsSquared.push_back(curSquare);
          denomRecords[i] = v;
          numerRecords[i] = numer;
        }
        else
        {
          // When not on the first step, can overwrite the previous stored values
          taus[i]          = tau;
          derivsSquared[i] = curSquare;
          denomRecords[i]  = v;
          numerRecords[i]  = numer;
        }

        paramsCopy[i] = currentParams.at(i);
      }
    }
    // AMSGrad method, similar to ADAM except for form of the step size denominator
    else if (flavor.compare("AMSGrad") == 0)
    {
      app_log() << "Using AMSGrad" << std::endl;


      for (int i = 0; i < numParams; i++)
      {
        double curSquare = std::pow(curDerivSet.at(i), 2);
        double beta1     = .9;
        double beta2     = .99;
        if (descent_num_ == 0)
        {
          numerRecords.push_back(0);
          denomRecords.push_back(0);
        }

        numer = beta1 * numerRecords[i] + (1 - beta1) * curDerivSet[i];
        v     = beta2 * denomRecords[i] + (1 - beta2) * curSquare;
        v     = std::max(denomRecords[i], v);

        denom    = std::sqrt(v) + epsilon;
        type_Eta = this->setStepSize(i);
        tau      = type_Eta / denom;

        currentParams.at(i) = currentParams.at(i) - tau * numer;

        if (taus.size() < numParams)
        {
          // For the first optimization step, need to add to the vectors
          taus.push_back(tau);
          derivsSquared.push_back(curSquare);
          denomRecords[i] = v;
          numerRecords[i] = numer;
        }
        else
        {
          // When not on the first step, can overwrite the previous stored values
          taus[i]          = tau;
          derivsSquared[i] = curSquare;
          denomRecords[i]  = v;
          numerRecords[i]  = numer;
        }

        paramsCopy[i] = currentParams.at(i);
      }
    }
  }


  descent_num_++;
}

// Helper method for setting step size according parameter type.
double DescentEngine::setStepSize(int i)
{
  double type_eta;


  std::string name = engineParamNames[i];


  int type = engineParamTypes[i];

  //Step sizes are assigned according to parameter type identified from the variable name.
  //Other parameter types could be added to this section as other wave function ansatzes are developed.
  if ((name.find("uu") != std::string::npos) || (name.find("ud") != std::string::npos))
  {
    type_eta = TJF_2Body_eta;
  }
  //If parameter name doesn't have "uu" or "ud" in it and is of type 1, assume it is a 1 body Jastrow parameter.
  else if (type == 1)
  {
    type_eta = TJF_1Body_eta;
  }
  else if (name.find("F_") != std::string::npos)
  {
    type_eta = F_eta;
  }
  else if (name.find("CIcoeff_") != std::string::npos || name.find("CSFcoeff_") != std::string::npos)
  {
    type_eta = CI_eta;
  }
  else if (name.find("orb_rot_") != std::string::npos)
  {
    type_eta = Orb_eta;
  }
  else if (name.find("g") != std::string::npos)
  {
    //Gaussian parameters are rarely optimized in practice but the descent code allows for it.
    type_eta = Gauss_eta;
  }
  else
  {
    //If there is some other parameter type that isn't in one of the categories with a default/input, use a conservative default step size.
    type_eta = .001;
  }

  if (ramp_eta && descent_num_ < ramp_num)
  {
    type_eta = type_eta * (descent_num_ + 1) / ramp_num;
  }

  return type_eta;
}

//Method for retrieving parameter values, names, and types from the VariableSet before the first descent optimization step
void DescentEngine::setupUpdate(const optimize::VariableSet& myVars)
{
  numParams = myVars.size();
  for (int i = 0; i < numParams; i++)
  {
    engineParamNames.push_back(myVars.name(i));
    engineParamTypes.push_back(myVars.getType(i));
    paramsCopy.push_back(myVars[i]);
    currentParams.push_back(myVars[i]);
    paramsForDiff.push_back(myVars[i]);
  }
}

// Helper method for storing vectors of parameter differences over the course of
// a descent optimization for use in BLM steps of the hybrid method
void DescentEngine::storeVectors(std::vector<double>& currentParams)
{
  std::vector<double> rowVec(currentParams.size(), 0.0);

  // Take difference between current parameter values and the values from 20
  // iterations before (in the case descent_len = 100) to be stored as input to BLM.
  // The current parameter values are then copied to paramsForDiff to be used
  // another 20 iterations later.
  for (int i = 0; i < currentParams.size(); i++)
  {
    rowVec[i]        = currentParams[i] - paramsForDiff[i];
    paramsForDiff[i] = currentParams[i];
  }

  // If on first store of descent section, clear anything that was in vector
  if (store_count == 0)
  {
    hybridBLM_Input.clear();
    hybridBLM_Input.push_back(rowVec);
  }
  else
  {
    hybridBLM_Input.push_back(rowVec);
  }

  for (int i = 0; i < hybridBLM_Input.size(); i++)
  {
    std::string entry = "";
    for (int j = 0; j < hybridBLM_Input.at(i).size(); j++)
    {
      entry = entry + std::to_string(hybridBLM_Input.at(i).at(j)) + ",";
    }
    app_log() << "Stored Vector: " << entry << std::endl;
  }
  store_count++;
}

} // namespace qmcplusplus
