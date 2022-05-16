#ifndef HYDROBRICKS_FLUX_TO_ATMOSPHERE_H
#define HYDROBRICKS_FLUX_TO_ATMOSPHERE_H

#include "Includes.h"
#include "Flux.h"

class FluxToAtmosphere : public Flux {
  public:
    explicit FluxToAtmosphere();

    /**
     * @copydoc Flux::IsOk()
     */
    bool IsOk() override;

    /**
     * @copydoc Flux::GetAmount()
     */
    double GetAmount() override;

  protected:

  private:
};

#endif  // HYDROBRICKS_FLUX_TO_ATMOSPHERE_H
