

#include <vector>
#include <string>

#include "QMCDrivers/Optimizers/HybridEngine.h"
#include "OhmmsData/ParameterSet.h"
#include "Message/CommOperators.h"
#include "OhmmsData/XMLParsingString.h"


namespace qmcplusplus
{
HybridEngine::HybridEngine(Communicate* comm, const xmlNodePtr cur) : myComm(comm)
{
  step_num_ = -1;
  processXML(cur);
}


bool HybridEngine::processXML(const xmlNodePtr opt_xml)
{
  opt_methods_.clear();
  saved_xml_opt_methods_.clear();
  num_updates_opt_methods_.clear();

  xmlNodePtr cur = opt_xml->children;
  while (cur != NULL)
  {
    std::string cname((const char*)(cur->name));
    if (cname == "optimizer")
    {
      std::string children_MinMethod;
      ParameterSet m_param;
      m_param.add(children_MinMethod, "MinMethod", "string");
      m_param.put(cur);

      if (children_MinMethod.empty())
        throw std::runtime_error("MinMethod must be given!\n");
      XMLAttrString updates_string(cur, "num_updates");
      app_log() << "HybridEngine saved MinMethod " << children_MinMethod << " num_updates = " << updates_string
                << std::endl;
      auto iter = OptimizerNames.find(children_MinMethod);
      if (iter == OptimizerNames.end())
        throw std::runtime_error("Unknown MinMethod!\n");
      opt_methods_.push_back(iter->second);
      saved_xml_opt_methods_.push_back(cur);
      num_updates_opt_methods_.push_back(std::stoi(updates_string));
    }
    cur = cur->next;
  }

  if (saved_xml_opt_methods_.size() != 2)
    throw std::runtime_error("MinMethod hybrid needs two optimizer input blocks!\n");

  return true;
}

xmlNodePtr HybridEngine::getSelectedXML()
{
  step_num_++;

  return saved_xml_opt_methods_[identifyMethodIndex()];
}

bool HybridEngine::queryStore(int store_num, const OptimizerType method) const
{
  bool store = false;

  int idx = 0;

  auto iter = std::find(opt_methods_.begin(), opt_methods_.end(), method);
  if (iter == opt_methods_.end())
    throw std::runtime_error("Unknown MinMethod!\n");
  else
    idx = std::distance(opt_methods_.begin(), iter);


  const int totMicroIt = std::accumulate(num_updates_opt_methods_.begin(), num_updates_opt_methods_.end(), 0);
  const int pos        = step_num_ % totMicroIt;
  int interval         = num_updates_opt_methods_[idx] / store_num;

  if (interval == 0)
  {
    app_log()
        << "Requested Number of Stored Vectors greater than number of descent steps. Storing a vector on each step."
        << std::endl;
    interval = 1;
  }
  if ((pos + 1) % interval == 0)
  {
    store = true;
  }

  return store;
}

int HybridEngine::identifyMethodIndex() const
{
  const int totMicroIt = std::accumulate(num_updates_opt_methods_.begin(), num_updates_opt_methods_.end(), 0);
  const int pos        = step_num_ % totMicroIt;

  int runSum    = 0;
  int selectIdx = 0;

  //Compare pos to running sum of microiterations of different methods to determine which method is being used
  for (int i = 0; i < num_updates_opt_methods_.size(); i++)
  {
    runSum += num_updates_opt_methods_[i];
    if (runSum > pos)
    {
      selectIdx = i;
      break;
    }
  }

  return selectIdx;
}

} // namespace qmcplusplus
