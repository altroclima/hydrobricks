#include "Parameter.h"
#include "SettingsModel.h"

SettingsModel::SettingsModel()
    : m_selectedStructure(nullptr),
      m_selectedBrick(nullptr),
      m_selectedProcess(nullptr),
      m_selectedSplitter(nullptr)
{
    ModelStructure initialStructure;
    initialStructure.id = 1;
    m_modelStructures.push_back(initialStructure);
    m_selectedStructure = &m_modelStructures[0];
}

SettingsModel::~SettingsModel() {
    for (auto &modelStructure : m_modelStructures) {
        for (auto &brick : modelStructure.bricks) {
            for (auto &parameter : brick.parameters) {
                wxDELETE(parameter);
            }
        }
        for (auto &splitter : modelStructure.splitters) {
            for (auto &parameter : splitter.parameters) {
                wxDELETE(parameter);
            }
        }
    }
}

void SettingsModel::SetSolver(const wxString &solverName) {
    m_solver.name = solverName;
}

void SettingsModel::SetTimer(const wxString &start, const wxString &end, int timeStep, const wxString &timeStepUnit) {
    m_timer.start = start;
    m_timer.end = end;
    m_timer.timeStep = timeStep;
    m_timer.timeStepUnit = timeStepUnit;
}

void SettingsModel::AddBrick(const wxString &name, const wxString &type) {
    wxASSERT(m_selectedStructure);

    BrickSettings brick;
    brick.name = name;
    brick.type = type;

    m_selectedStructure->bricks.push_back(brick);
    m_selectedBrick = &m_selectedStructure->bricks[m_selectedStructure->bricks.size() - 1];
}

void SettingsModel::AddParameterToCurrentBrick(const wxString &name, float value, const wxString &type) {
    wxASSERT(m_selectedBrick);

    if (!type.IsSameAs("Constant")) {
        throw NotImplemented();
    }

    auto parameter = new Parameter(name, value);

    m_selectedBrick->parameters.push_back(parameter);
}

void SettingsModel::AddForcingToCurrentBrick(const wxString &name) {
    wxASSERT(m_selectedBrick);

    if (name.IsSameAs("Precipitation", false)) {
        m_selectedBrick->forcing.push_back(Precipitation);
    } else if (name.IsSameAs("Temperature", false)) {
        m_selectedBrick->forcing.push_back(Temperature);
    } else {
        throw InvalidArgument(_("The provided forcing is not yet supported."));
    }
}

void SettingsModel::AddProcessToCurrentBrick(const wxString &name, const wxString &type) {
    wxASSERT(m_selectedBrick);

    ProcessSettings processSettings;
    processSettings.name = name;
    processSettings.type = type;
    m_selectedBrick->processes.push_back(processSettings);

    m_selectedProcess = &m_selectedBrick->processes[m_selectedBrick->processes.size() - 1];
}

void SettingsModel::AddParameterToCurrentProcess(const wxString &name, float value, const wxString &type) {
    wxASSERT(m_selectedProcess);

    if (!type.IsSameAs("Constant")) {
        throw NotImplemented();
    }

    auto parameter = new Parameter(name, value);

    m_selectedProcess->parameters.push_back(parameter);
}

void SettingsModel::AddForcingToCurrentProcess(const wxString &name) {
    wxASSERT(m_selectedProcess);

    if (name.IsSameAs("Precipitation", false)) {
        m_selectedProcess->forcing.push_back(Precipitation);
    } else if (name.IsSameAs("PET", false)) {
        m_selectedProcess->forcing.push_back(PET);
    } else if (name.IsSameAs("Temperature", false)) {
        m_selectedProcess->forcing.push_back(Temperature);
    } else {
        throw InvalidArgument(_("The provided forcing is not yet supported."));
    }
}

void SettingsModel::AddOutputToCurrentProcess(const wxString &target) {
    wxASSERT(m_selectedProcess);

    OutputSettings outputSettings;
    outputSettings.target = target;
    m_selectedProcess->outputs.push_back(outputSettings);
}

void SettingsModel::AddSplitter(const wxString &name, const wxString &type) {
    wxASSERT(m_selectedStructure);

    SplitterSettings splitter;
    splitter.name = name;
    splitter.type = type;

    m_selectedStructure->splitters.push_back(splitter);
    m_selectedSplitter = &m_selectedStructure->splitters[m_selectedStructure->splitters.size() - 1];
}

void SettingsModel::AddParameterToCurrentSplitter(const wxString &name, float value, const wxString &type) {
    wxASSERT(m_selectedSplitter);

    if (!type.IsSameAs("Constant")) {
        throw NotImplemented();
    }

    auto parameter = new Parameter(name, value);

    m_selectedSplitter->parameters.push_back(parameter);
}

void SettingsModel::AddForcingToCurrentSplitter(const wxString &name) {
    wxASSERT(m_selectedSplitter);

    if (name.IsSameAs("Precipitation", false)) {
        m_selectedSplitter->forcing.push_back(Precipitation);
    } else if (name.IsSameAs("Temperature", false)) {
        m_selectedSplitter->forcing.push_back(Temperature);
    } else {
        throw InvalidArgument(_("The provided forcing is not yet supported."));
    }
}

void SettingsModel::AddOutputToCurrentSplitter(const wxString &target) {
    wxASSERT(m_selectedSplitter);

    OutputSettings outputSettings;
    outputSettings.target = target;
    m_selectedSplitter->outputs.push_back(outputSettings);
}

void SettingsModel::AddLoggingToItem(const wxString& itemName) {
    wxASSERT(m_selectedStructure);
    m_selectedStructure->logItems.push_back(itemName);
}

void SettingsModel::AddLoggingToCurrentBrick(const wxString& itemName) {
    wxASSERT(m_selectedBrick);
    m_selectedBrick->logItems.push_back(itemName);
}

void SettingsModel::AddLoggingToCurrentProcess(const wxString& itemName) {
    wxASSERT(m_selectedProcess);
    m_selectedProcess->logItems.push_back(itemName);
}

void SettingsModel::AddLoggingToCurrentSplitter(const wxString& itemName) {
    wxASSERT(m_selectedSplitter);
    m_selectedSplitter->logItems.push_back(itemName);
}

bool SettingsModel::SelectStructure(int id) {
    for (auto &modelStructure : m_modelStructures) {
        if (modelStructure.id == id) {
            m_selectedStructure = &modelStructure;
            if (m_selectedStructure->bricks.empty()) {
                m_selectedBrick = nullptr;
            } else {
                m_selectedBrick = &m_selectedStructure->bricks[0];
            }
            return true;
        }
    }

    return false;
}

void SettingsModel::SelectBrick(int index) {
    wxASSERT(m_selectedStructure);
    wxASSERT(m_modelStructures.size() == 1);

    m_selectedBrick = &m_selectedStructure->bricks[index];
}

void SettingsModel::SelectProcess(int index) {
    wxASSERT(m_selectedBrick);
    wxASSERT(m_selectedBrick->processes.size() >= 1);

    m_selectedProcess = &m_selectedBrick->processes[index];
}

void SettingsModel::SelectSplitter(int index) {
    wxASSERT(m_selectedStructure);
    wxASSERT(m_modelStructures.size() == 1);

    m_selectedSplitter = &m_selectedStructure->splitters[index];
}

vecStr SettingsModel::GetHydroUnitLogLabels() {
    wxASSERT(m_selectedStructure);
    wxASSERT(m_modelStructures.size() == 1);

    vecStr logNames;

    for (auto &modelStructure : m_modelStructures) {
        for (auto &brick : modelStructure.bricks) {
            logNames.insert(logNames.end(), brick.logItems.begin(), brick.logItems.end());
            for (auto &process : brick.processes) {
                logNames.insert(logNames.end(), process.logItems.begin(), process.logItems.end());
            }
        }
        for (auto &splitter : modelStructure.splitters) {
            logNames.insert(logNames.end(), splitter.logItems.begin(), splitter.logItems.end());
        }
    }

    return logNames;
}

vecStr SettingsModel::GetAggregatedLogLabels() {
    wxASSERT(m_selectedStructure);
    wxASSERT(m_modelStructures.size() == 1);

    return m_selectedStructure->logItems;
}
