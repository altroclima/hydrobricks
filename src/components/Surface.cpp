#include "Surface.h"

Surface::Surface(HydroUnit *hydroUnit)
    : Brick(hydroUnit),
      m_waterHeight(0)
{}

void Surface::AssignParameters(const BrickSettings &brickSettings) {
    Brick::AssignParameters(brickSettings);
}

bool Surface::IsOk() {
    if (m_hydroUnit == nullptr) {
        wxLogError(_("The surface is not attached to a hydro unit."));
        return false;
    }
    if (m_inputs.empty()) {
        wxLogError(_("The surface is not attached to inputs."));
        return false;
    }
    for (auto process : m_processes) {
        if (!process->IsOk()) {
            return false;
        }
    }

    return true;
}

vecDouble Surface::ComputeOutputs() {
    return {};
}