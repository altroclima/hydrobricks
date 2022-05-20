#ifndef HYDROBRICKS_VEGETATION_H
#define HYDROBRICKS_VEGETATION_H

#include "SurfaceComponent.h"
#include "Includes.h"

class Vegetation : public SurfaceComponent {
  public:
    Vegetation(HydroUnit *hydroUnit);

    /**
     * @copydoc Brick::AssignParameters()
     */
    void AssignParameters(const BrickSettings &brickSettings) override;

    void ApplyConstraints(double timeStep) override;

    void Finalize() override;

  protected:

  private:
};

#endif  // HYDROBRICKS_VEGETATION_H
