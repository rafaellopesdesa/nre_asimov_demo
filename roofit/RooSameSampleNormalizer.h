#ifndef OFFSHELL_ROOSAMESAMPLENORMALIZER_H
#define OFFSHELL_ROOSAMESAMPLENORMALIZER_H

#include <RooAbsReal.h>
#include <RooListProxy.h>
#include <array>
#include <vector>

// Integral of rhat * PiecewiseInterpolation(code=5), with fixed u*rhat.
// Only nuisance parameters are RooFit servers: this is one shared scalar,
// independent of the currently evaluated event and of the Asimov weights.
class RooSameSampleNormalizer final : public RooAbsReal {
public:
  RooSameSampleNormalizer() = default;
  RooSameSampleNormalizer(const char *name, const char *title,
                          const RooArgList &parameters,
                          const std::vector<double> &measure,
                          const std::vector<double> &low,
                          const std::vector<double> &high);
  RooSameSampleNormalizer(const RooSameSampleNormalizer &other, const char *name = nullptr);
  TObject *clone(const char *name) const override { return new RooSameSampleNormalizer(*this, name); }
  double directValue() const; // Independent, uncached ROOT interpolation for validation.
protected:
  double evaluate() const override;
private:
  RooListProxy parameters_{"parameters", "nuisance parameters", this};
  std::vector<double> measure_, low_, high_;
  mutable std::vector<std::array<double,8>> polynomial_; //! Reconstructed optimization cache.
  mutable std::vector<double> previous_, factors_, product_; //! Transient numerical cache.
  mutable unsigned updates_ = 0; //!
  void initializeCache() const;
  ClassDefOverride(RooSameSampleNormalizer, 1)
};
#endif
