#include "ProcessMeltRadiation.h"

#include "Brick.h"
#include "WaterContainer.h"

ProcessMeltRadiation::ProcessMeltRadiation(WaterContainer* container)
    : ProcessMelt(container),
      m_temperature(nullptr),
	  m_meltFactor(nullptr),
      m_meltingTemperature(nullptr),
	  m_radiationCoefficient(nullptr),
	  m_potentialClearSkyDirectSolarRadiation(nullptr) {}

bool ProcessMeltRadiation::IsOk() {
    if (!ProcessMelt::IsOk()) {
        return false;
    }
    if (m_temperature == nullptr) {
        return false;
    }
    if (m_meltFactor == nullptr) {
        return false;
    }
    if (m_meltingTemperature == nullptr) {
        return false;
    }
    if (m_radiationCoefficient == nullptr) {
        return false;
    }
    if (m_potentialClearSkyDirectSolarRadiation == nullptr) {
        return false;
    }

    return true;
}

void ProcessMeltRadiation::SetParameters(const ProcessSettings& processSettings) {
    Process::SetParameters(processSettings);
    m_meltFactor = GetParameterValuePointer(processSettings, "melt_factor");
    m_meltingTemperature = GetParameterValuePointer(processSettings, "melting_temperature");
    m_radiationCoefficient = GetParameterValuePointer(processSettings, "radiation_coefficient");
}

void ProcessMeltRadiation::AttachForcing(Forcing* forcing) {
    if (forcing->GetType() == Temperature) {
        m_temperature = forcing;
    } else if (forcing->GetType() == Radiation) {
    	m_potentialClearSkyDirectSolarRadiation = forcing;
    } else {
        throw InvalidArgument("Forcing must be of type Temperature or Radiation");
    }
}

vecDouble ProcessMeltRadiation::GetRates() {
    if (!m_container->ContentAccessible()) {
        return {0};
    }

    double melt = 0;
    if (m_temperature->GetValue() >= *m_meltingTemperature) {
        melt = (m_temperature->GetValue() - *m_meltingTemperature) * (*m_meltFactor + *m_radiationCoefficient * m_potentialClearSkyDirectSolarRadiation->GetValue());
    }

    return {melt};
}
