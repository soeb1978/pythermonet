# ✅ Refactoring & Validation TODO Roadmap for `pythermonet`

This roadmap outlines the structured steps for refactoring, input validation, and preparing the codebase for robustness and maintainability.

---

## ✳️ Phase 1: Refactor Core Logic and File Structure

🎯 **Goal:** Cleanly separate responsibilities and get all existing examples working with the new structure.

- [ ] Split large logic blocks into clean, modular functions
  - [ ] Separate file reading from data parsing
  - [ ] Move raw input parsing into `io/aggregated_load.py` (and similar for other types)
  - [ ] Move all domain models into `models/`
  - [ ] Ensure `main.py` in each example is minimal and readable

- [ ] Define a clear flow per data type:
      File → IO function → Input dataclass → parse → Model → Simulation


- [ ] Avoid premature validation or fallback logic
- [ ] Hardcode column names if needed — just get the current examples working

---

## ✳️ Phase 2: Input Validation Pass

🎯 **Goal:** Ensure missing or malformed input is caught early and consistently.

- [ ] Column validation
- [ ] Define expected columns for each file type
- [ ] Raise `ValueError` if any are missing

- [ ] Value checks and defaults
- [ ] Add `safe_get()` helper for handling `NaN` or missing values
- [ ] Apply safe defaults (e.g., `COP = 1.0`, `load = 0.0`)

- [ ] Logging fallback/default behavior
- [ ] Use `logger.warning(...)` if fallback values are used
- [ ] Set up `logging_config.py` and initialize logging from `main.py`

---

## ✳️ Phase 3: Output & Runtime Validation (Optional)

🎯 **Goal:** Catch invalid calculations and clearly communicate problems.

- [ ] Validate core model inputs
- [ ] Add sanity checks (e.g. `assert cop > 1.0`)
- [ ] Raise/log errors for invalid config values

- [ ] Optional: Validate model outputs
- [ ] Check that values like `Qdim_H` fall in expected ranges
- [ ] Warn for suspicious results (e.g. negative loads)

---

## ✳️ Phase 4: Testing and Edge Case Handling

🎯 **Goal:** Ensure system handles both good and bad input gracefully.

- [ ] Add minimal test coverage (e.g. `tests/test_io.py`)
- [ ] Load with complete data
- [ ] Load with missing optional fields
- [ ] Load with malformed/missing files
- [ ] Load with extra/unexpected fields

- [ ] Optional: Log input summary or report
- [ ] "Input file X loaded with fallback for Y"
- [ ] Summary of warnings at end of run

---

## ✳️ Phase 5: General Polish and UX

🎯 **Goal:** Prepare the system for new contributors and non-dev users.

- [ ] Improve error messages
- [ ] Use clear, user-facing language
- [ ] Include file paths, column names, and hints for fixing

- [ ] Update documentation / README
- [ ] Describe expected input formats
- [ ] Document fallback/default logic
- [ ] Explain how logging works and where logs go

---

## 🧩 Bonus Ideas (for future work)

- [ ] Add JSON schema validation for config files
- [ ] Support environment variables or CLI overrides
- [ ] Separate official examples from experimental ones

---

## ✅ Immediate Priority

1. [ ] Finish splitting/refactoring logic
2. [ ] Get existing examples working again with new structure
3. [ ] THEN: Revisit and improve input validation (Phase 2)

---


## recommended structure
      pythermonet/
      ├── core/                # Core physical and mathematical logic
      │   ├── physics.py       # Functions like network_load_from_COP, heat transfer, Reynolds, etc.
      │   ├── hydraulics.py    # Pressure drop, flow rate, etc.
      │   ├── thermal.py       # Rb, g-function utilities, temperature calc
      │   └── utils.py         # Small helpers, generic math/logical tools
      │
      ├── domain/              # Cleaned domain models & logic
      │   ├── models.py        # All dataclasses: Brine, AggregatedLoad, HeatPump, etc.
      │   ├── heatpump.py      # Heat pump-specific logic (load conversion, dimensioning)
      │   ├── source.py        # BHE and HHE source config and logic
      │   └── network.py       # Thermonet + pipe design logic
      │
      ├── io/                  # File parsing and I/O (no logic!)
      │   ├── load_csv.py      # read_topology, read_heatpump_data, etc.
      │   └── config.py        # JSON/YAML config parsing
      │
      ├── engine/              # High-level workflows and orchestrators
      │   ├── dimensioning.py  # run_full_dimensioning and related
      │   ├── pipe_sizing.py   # run_pipedimensioning
      │   ├── source_sizing.py # run_sourcedimensioning
      │   └── results.py       # Reporting or printing (can move to CLI layer later)
      │
      ├── cli/                 # If you add CLI or UI interfaces later
      │
      ├── tests/
      │   ├── unit/
      │   ├── integration/
      │   └── regression/


## Current progress.
I am working my way into the reading functions for the topology in the `src\pythermonet\io\topology.py` file.
 @@@### current progress I have mainly pasted and cleaned the function
 for handling the old version of the thermonet class and undimensioned
 topology. 
 Now I nedd to make the logic for the dimensions topology which I want
  to save to a new class that I have to create, I will do this as it 
 then serves as the entrance for the data whether it is from the .dat 
 file or from our json format.