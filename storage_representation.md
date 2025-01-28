# Application
The different storage representation methods can be applied by changing two attributes in the config's analysis as following:

```{
  "analysis": {
    "time_series_aggregation": {"storageRepresentationMethod": "ZEN-garden",
                                "hoursPerPeriod": 1}
  }
}
```
Besides the default ```storageRepresentatationMethod```  ```ZEN-garden```, ```kotzur```and ```gabrielli``` can be chosen. To match the number of used time steps per aggregation period, the ```hoursPerPeriod```attribute
can be changed from its default value 1 to 24 (or other numbers).


# Implementation
## storage_technology.py
### construct_vars()
Addtional variables for Kotzur

### constraint_storage_level_max()
upper bound formulation for Kotzur as well as lower bound for Kotur, since the superposed storage level cannot be constrainted directly as there doesn't exist an extra variable for the overall storage level.
### constraint_couple_storage_level()
kotzur intra as well as inter storage level coupling. set storage_level_intra at first time steps of intra period to zero.

## time_series_aggregation.py
### calculate_time_steps_storage_level()
calculate needed storage time steps for all methods

### set_time_attributes()
adapt time step sets accordingly to ```hoursPerPeriod```

## energy_system.py
additonal time step sets for kotzur

## time_steps.py
add inter and intra time steps

## default_config.py
inter and intra storage time steps. addtional ```time_series_aggregation``` attributes to chose representation method.

## element.py
several new methods to formulate the kotzur constraints

## multi_hdf_loader.py and solution_loader.py
adaptions sucht that kotzur results can be loaded.
