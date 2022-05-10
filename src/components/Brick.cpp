#include "Brick.h"

#include "Glacier.h"
#include "HydroUnit.h"
#include "Snowpack.h"
#include "Storage.h"
#include "Surface.h"

Brick::Brick(HydroUnit *hydroUnit)
    : m_needsSolver(true),
      m_content(0),
      m_capacity(nullptr),
      m_hydroUnit(hydroUnit)
{
    if (hydroUnit) {
        hydroUnit->AddBrick(this);
    }
}

Brick* Brick::Factory(const BrickSettings &brickSettings, HydroUnit* unit) {
    if (brickSettings.type.IsSameAs("Storage")) {
        auto brick = new Storage(unit);
        brick->AssignParameters(brickSettings);
        return brick;
    } else if (brickSettings.type.IsSameAs("Surface")) {
        auto brick = new Surface(unit);
        brick->AssignParameters(brickSettings);
        return brick;
    } else if (brickSettings.type.IsSameAs("Glacier")) {
        auto brick = new Glacier(unit);
        brick->AssignParameters(brickSettings);
        return brick;
    } else if (brickSettings.type.IsSameAs("Snowpack")) {
        auto brick = new Snowpack(unit);
        brick->AssignParameters(brickSettings);
        return brick;
    } else {
        wxLogError(_("Brick type '%s' not recognized."), brickSettings.type);
    }

    return nullptr;
}

void Brick::AssignParameters(const BrickSettings &brickSettings) {
    if (HasParameter(brickSettings, "capacity")) {
        m_capacity = GetParameterValuePointer(brickSettings, "capacity");
    }
}

bool Brick::HasParameter(const BrickSettings &brickSettings, const wxString &name) {
    for (auto parameter: brickSettings.parameters) {
        if (parameter->GetName().IsSameAs(name, false)) {
            return true;
        }
    }

    return false;
}

float* Brick::GetParameterValuePointer(const BrickSettings &brickSettings, const wxString &name) {
    for (auto parameter: brickSettings.parameters) {
        if (parameter->GetName().IsSameAs(name, false)) {
            wxASSERT(parameter->GetValuePointer());
            parameter->SetAsLinked();
            return parameter->GetValuePointer();
        }
    }

    throw MissingParameter(wxString::Format(_("The parameter '%s' could not be found."), name));
}

Process* Brick::GetProcess(int index) {
    wxASSERT(m_processes.size() > index);
    wxASSERT(m_processes[index]);

    return m_processes[index];
}

void Brick::SubtractAmount(double change) {
    m_content -= change;
    wxASSERT(m_content >= 0);
}

void Brick::AddAmount(double change) {
    m_content += change;
    wxASSERT(m_content >= 0);
}

bool Brick::Compute() {
    /*
    // Sum inputs
    double qIn = SumIncomingFluxes();

    // Compute outputs and sum
    vecDouble qOuts = ComputeOutputs();
    double qOutTotal = GetOutputsSum(qOuts);

    // Compute change rate
    m_contentChangeRate = qIn - qOutTotal;

    // If we encounter a negative content, change outputs proportionally
    if (m_content + m_contentChangeRate < 0) {
        double diff = m_contentChangeRate - m_content;
        for (double &qOut : qOuts) {
            qOut -= diff * (qOut / qOutTotal);
        }
        qOutTotal = GetOutputsSum(qOuts);
        m_contentChangeRate = qIn - qOutTotal;
    }

    // In case of a maximum capacity, check that it does not get topped out
    if (HasMaximumCapacity() && m_content + m_contentChangeRate > *m_capacity) {
        wxLogError(_("Content capacity of the storage exceeded. It has to be handled before."));
        return false;
    }

    for (int i = 0; i < qOuts.size(); ++i) {
        m_outputs[i]->SetAmount(qOuts[i]);
    }
*/
    return true;
}

double Brick::SumIncomingFluxes() {
    wxASSERT(!m_inputs.empty());

    double sum = 0;
    for (auto & input : m_inputs) {
        sum += input->GetAmount();
    }

    return sum;
}

void Brick::UpdateContentFromInputs() {
    m_content += SumIncomingFluxes();
}

double Brick::GetOutputsSum(vecDouble &qOuts) {
    double qOutTotal = 0;
    for (auto &qOut : qOuts) {
        qOutTotal += qOut;
    }

    return qOutTotal;
}

vecDoublePt Brick::GetStateVariables() {
    return vecDoublePt {&m_content};
}

vecDoublePt Brick::GetStateVariablesFromProcesses() {
    vecDoublePt values;
    for (auto const &process: m_processes) {
        vecDoublePt processValues = process->GetStateVariables();

        if (processValues.empty()) {
            continue;
        }
        for (auto const &value : processValues) {
            values.push_back(value);
        }
    }

    return values;
}

int Brick::GetProcessesConnectionsNb() {
    int counter = 0;

    for (auto const &process: m_processes) {
        counter += process->GetConnectionsNb();
    }

    return counter;
}

double* Brick::GetBaseValuePointer(const wxString& name) {
    if (name.IsSameAs("content")) {
        return &m_content;
    }

    return nullptr;
}

double* Brick::GetValuePointer(const wxString&) {
    return nullptr;
}