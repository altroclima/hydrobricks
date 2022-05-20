#ifndef HYDROBRICKS_GENERIC_SURFACE_H
#define HYDROBRICKS_GENERIC_SURFACE_H

#include "SurfaceComponent.h"
#include "Includes.h"

class GenericSurface : public SurfaceComponent {
  public:
    GenericSurface(HydroUnit *hydroUnit);

    /**
     * @copydoc Brick::AssignParameters()
     */
    void AssignParameters(const BrickSettings &brickSettings) override;

    void ApplyConstraints(double timeStep) override;

    void Finalize() override;

  protected:

  private:
};

#endif  // HYDROBRICKS_GENERIC_SURFACE_H
