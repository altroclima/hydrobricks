#include "SplitterRain.h"

SplitterRain::SplitterRain()
    : Splitter(),
      m_precipitation(nullptr) {}

bool SplitterRain::IsOk() {
    if (m_outputs.size() != 1) {
        wxLogError(_("SplitterRain should have 1 output."));
        return false;
    }

    return true;
}

void SplitterRain::SetParameters(const SplitterSettings&) {
    //
}

void SplitterRain::AttachForcing(Forcing* forcing) {
    if (forcing->GetType() == Precipitation) {
        m_precipitation = forcing;
    } else {
        throw InvalidArgument("Forcing must be of type Precipitation");
    }
}

double* SplitterRain::GetValuePointer(const string& name) {
    if (name == "rain") {
        return m_outputs[0]->GetAmountPointer();
    }

    return nullptr;
}

void SplitterRain::Compute() {
    m_outputs[0]->UpdateFlux(m_precipitation->GetValue());
}
