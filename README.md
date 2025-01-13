# Technology - and Location-Specific Cost of Capital in ZEN-garden
This repository contains the code and data used to calculate the technology- and location-specific cost of capital (modeled through WACC) in the ZEN-garden optimization framework. 
This enables a more precise representation of the financing costs of technologies in the modeling framework. 
Following adjustments must be made to use variable CoC:
1. Select the variable CoC in the `analysis.json` file.
2. Add the WACC input parameter to the technologies' attributes in your input dataset.

## How is it modelled? How does it effect the model outcome?
We reflect the influence of the cost of capital on the investment costs of technologies by making 
the annuity factor dependent on the weighted average cost of capital (WACC). The annuity factor annualizes the 
investment costs (including capital costs) of a technology over its lifetime and, thus, determines the annual CAPEX costs. 
With a rising WACC, the annuity factor increases, leading to higher annual CAPEX costs.

## How to select variable CoC modeling approach
To select a variable CoC, you need make following adjustments to the analysis settings summarized in `analysis.json`:
```bash
"analysis": {"variable_CoC": true}
```

## How to add WACC input parameter to the technologies' attributes
For all technologies (i.e. conversion, storage, and transport technologies) the `attributes.json` file now requires an 
additional input parameter `WACC` to be added. The `WACC` parameter is the weighted average cost of capital for the technology.
If the `WACC` also varies by location (e.g. country), country-technology specific WACC values may be provided through a 
csv file called `WACC.csv` in the subfolder of the respective technology. The `WACC.csv` file should have the following columns:
- `node`: The nodes for which the WACC value is provided.
- `WACC`: WACC values for the respective nodes.

Consequently, WACC values may be implemented that vary by technology and location.

## Changes to current ZEN-garden version
The changes to the current version of ZEN-garden are made in the following files:

##### `element.py`:
- The function `get_discount_factor(self, calling_class, get_WACC=False)` is added. This function either returns the yearly series of discount factors for deriving the Net Present Costs of technologies, carriers, and the system, or the (location-technology-specific) WACC value(s) for calculating the annuity factor.

##### `technology.py`:
- Input parameter `WACC` is added to `params` and `store_input_data` functions.
- The function/technology rule `constraint_cost_capex_yearly(self)` is modified such that the discount rate used in the calculation of the annuity factor is obtained from the `get_discount_factor` function and corresponds to the (location-technology-specific) WACC. 

## Dataset example and plot script
In the folder CoC_data you can find the dataset (`PI_CoC`) used for the semester project. 
The dataset contains (per default) the variable CoC values (also summarized in the csv file `CoC_completed.csv`) as well as the uniform CoC and policy scenarios for the semester project (defined in the `scenarios.json` file). 
The script `analyze_results.py` in the `CoC_data` folder can be used to recreate the plots of the semester project report (note `countries.zip` contains the shapefile data for plotting the EU). 

## Further comments

### Net Present Cost decomposition
As we first used the approach of discounting costs (i.e. NPC) with the variable CoC, instead of just modeling it through its influence on the annuity factor, the code here also 
decomposes the NPC, which was before only summarized on the total system level, into the individual technologies, carriers, and the system (emission overshoot) NPC components.
Thus, following variables are added:
- `net_present_cost_yearly_technology` in `technology.py` that captures the annual NPC of all technologies.
- `net_present_cost_yearly_carrier` in `carrier.py` that captures the annual NPC of all carriers.
- `net_present_cost_system` in `system.py` that captures the annual NPC of the system.
For the analysis of, for instance, LCOE values per country, the NPC decomposition is useful.
However, the NPC decomposition can also easily be discarded if not needed. For this, following things must be done:
- Remove the `net_present_cost_yearly_technology`, `net_present_cost_yearly_carrier`, and `net_present_cost_system` variables from the respective classes.
- Remove the rules `constraint_net_present_cost_carrier(self)` in `carrier.py`, `constraint_net_present_cost_technology()` in `technology.py`, and `constraint_net_present_cost_system(self)` in `energy_system.py`.
- Remove the rule `constraint_add_net_present_cost(self)` in `energy_system.py` and uncomment the rule `rules.constraint_net_present_cost()` in line 358 in `energy_system.py`.

If on the other hand the NPC decomposition is desired, functions that double certain functionalities (such as `constraint_cost_total`) can be removed instead and are labeled with a comment `#ToDo: can be removed for variable CoC`. (This might throw errors for the visualization platform though.)

### WACC calculation inside ZEN-garden
The code also includes a configuration that allows to calculate location-technology-specific WACC values inside ZEN-garden. This is done based on following formula based on https://www.nature.com/articles/s41560-024-01606-7,
```
WACC = debt_ratio * (1-tax_rate) * (interest_rate +technology_premium) + (1-debt_ratio) * (interest_rate + equity_margin + technology_premium)
```
where the `debt_ratio`, `tax_rate`, `interest_rate`, `equity_margin`, and `technology_premium` are additional input parameters of the technologies and energy system.
Following a table that summarizes the input parameters for this WACC calculation:

| Parameter | Description                          | Input class    | Can vary across            |
| --- |--------------------------------------|----------------|----------------------------|
| `debt_ratio` | Debt ratio of the technology         | `Technology`   | Technology,Time            |
| `tax_rate` | Corporate tax rate                   | `EnergySystem` | Location, Time             |
| `interest_rate` | Interest rate of location            | `EnergySystem` | Location, Time             |
| `equity_margin` | Equity margin of the country         | `EnergySystem` | Location, Time             |
| `technology_premium` | Technology premium of the technology | `Technology`    | Technology, Locatino, Time |

To use this WACC calculation, the following adjustments must be made to the analysis settings summarized in `analysis.json`:
```bash
"analysis": {"variable_CoC": true, "calculate_WACC": true}
```
