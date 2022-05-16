#ifndef HYDROBRICKS_HYDRO_UNIT_H
#define HYDROBRICKS_HYDRO_UNIT_H

#include "Brick.h"
#include "Forcing.h"
#include "Splitter.h"
#include "HydroUnitProperty.h"
#include "Includes.h"

class HydroUnit : public wxObject {
  public:
    enum Types {
        Distributed,
        SemiDistributed,
        Lumped,
        Undefined
    };

    HydroUnit(float area = UNDEFINED, Types type = Undefined);

    ~HydroUnit() override;

    void AddProperty(HydroUnitProperty* property);

    void AddBrick(Brick* brick);

    void AddSplitter(Splitter* splitter);

    bool HasForcing(VariableType type);

    void AddForcing(Forcing* forcing);

    Forcing* GetForcing(VariableType type);

    int GetBricksCount();

    int GetSplittersCount();

    Brick* GetBrick(int index);

    bool HasBrick(const wxString &name);

    Brick* GetBrick(const wxString &name);

    Splitter* GetSplitter(int index);

    bool HasSplitter(const wxString &name);

    Splitter* GetSplitter(const wxString &name);

    bool IsOk();

    Types GetType() {
        return m_type;
    }

    void SetId(long id) {
        m_id = id;
    }

    float GetArea() {
        return m_area;
    }

    long GetId() const {
        return m_id;
    }

  protected:
    Types m_type;
    long m_id;
    float m_area; // m2
    std::vector<HydroUnitProperty*> m_properties;
    std::vector<Brick*> m_bricks;
    std::vector<Splitter*> m_splitters;
    std::vector<Forcing*> m_forcing;

  private:
};

#endif
