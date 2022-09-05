#ifndef HYDROBRICKS_PARAMETERS_UPDATER_H
#define HYDROBRICKS_PARAMETERS_UPDATER_H

#include "Includes.h"
#include "ParameterVariable.h"

class ParametersUpdater : public wxObject {
  public:
    ParametersUpdater();

    ~ParametersUpdater() override = default;

    void AddParameterVariableYearly(ParameterVariableYearly* parameter);

    void AddParameterVariableMonthly(ParameterVariableMonthly* parameter);

    void AddParameterVariableDates(ParameterVariableDates* parameter);

    void DateUpdate(double date);

    double GetPreviousDate() {
        return m_previousDate;
    }

  protected:
    void ChangingYear(int year);

    void ChangingMonth(int month);

    void ChangingDate(double date);

  private:
    double m_previousDate;
    std::vector<ParameterVariableYearly*> m_parametersYearly;
    std::vector<ParameterVariableMonthly*> m_parametersMonthly;
    std::vector<ParameterVariableDates*> m_parametersDates;
};

#endif  // HYDROBRICKS_PARAMETERS_UPDATER_H
