.. _t_add_tech_carrier.t_add_tech_carrier:

########################################
Tutorial 6: Add a technology and carrier
########################################

This tutorial will guide you through the process of adding a new technology or carrier to your dataset.
This is a common task when you want to expand an existing dataset with new technologies or carriers.

This tutorial assumes that you have installed and run the example dataset
``5_multiple_time_steps_per_year`` as described in the tutorial :ref:`setup
instructions <tutorials_intro.setup>`.

.. _t_add_tech.t_add_tech:

Adding a new technology
=======================

To add a new technology to your dataset, you need to follow these steps:

Identify technology type
^^^^^^^^^^^^^^^^^^^^^^^^
Identify whether you are adding a conversion (or retrofitting) technology, a storage technology,
or a transport technology. This will determine where you need to add the new technology in the input data structure.

Create new folder
^^^^^^^^^^^^^^^^^
Add a new folder in the appropriate folder:

- For conversion technologies, add a new folder in ``set_technologies/conversion_technologies/``.
- For storage technologies, add a new folder in ``set_technologies/storage_technologies/``.
- For transport technologies, add a new folder in ``set_technologies/transport_technologies/``.
- For retrofitting technologies, add a new folder in ``set_technologies/conversion_technologies/retrofitting_technologies/``.

You can copy an existing folder and rename it to the name of your new technology.

Add/change ``attributes.json``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the new folder, edit the ``attributes.json`` file to include the attributes of your new technology.
First, you need to change the ``reference_carrier``
(and for conversion technologies also the ``input_carriers`` and ``output_carriers``).
These must be carriers that exist in your dataset (see :ref:`adding a new carrier <t_add_carrier.t_add_carrier>`).
Then, you can change the other attributes to reflect the characteristics of your new technology.

.. note::
    Make sure that all units are consistent with the rest of your dataset. Units are coupled through the
    units of the carriers. For example, if you define the ``reference_carrier`` as ``hydrogen`` and
    the unit of the capacity is ``tons/h``, then the unit of the hydrogen demand must be ``tons/h`` as well.

Add other input files
^^^^^^^^^^^^^^^^^^^^^

You can add other input files to further specify the characteristics of your new technology.
Refer to :ref:`input handling documentation<input_handling.overwrite_defaults>` for a detailed description
of how to overwrite default values.

Add technology to ``system.json``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a last step, you need to add the new technology to the ``system.json`` file.
Again, add it to the appropriate section depending on the type of technology you are adding.
If you do not add the new technology to ``system.json``, it will not be included in the model.

.. _t_add_carrier.t_add_carrier:

Adding a new carrier
====================

The steps to add a carrier are very similar to adding a new technology.

Create new folder
^^^^^^^^^^^^^^^^^
Add a new folder in the ``set_carriers/`` folder.
You can copy an existing folder and rename it to the name of your new carrier.
The name of the folder will be the name of the new carrier.

Add/change ``attributes.json``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the new folder, edit the ``attributes.json`` file to include the attributes of your new carrier.
Then, you can change the other attributes to reflect the characteristics of your new carrier.

.. note::
    Make sure that all units are consistent with the rest of your dataset. Units are coupled through the
    units of the carriers. For example, if you define the unit of the hydrogen demand as ``tons/h``,
    then the unit of the capacity of a technology using hydrogen as reference carrier must be ``tons/h`` as well.

Add other input files
^^^^^^^^^^^^^^^^^^^^^

You can add other input files to further specify the characteristics of your new carrier.
Refer to :ref:`input handling documentation<input_handling.overwrite_defaults>` for a detailed description
of how to overwrite default values.

.. note::

    You do not need to add the new carrier to ``system.json``.
    All carriers implied by the technologies in ``system.json`` are automatically included in the model.


.. _t_add_tech_carrier.exercise:

Example Exercise
================

Let us go through an example of adding a new technology and carrier to the dataset
``5_multiple_time_steps_per_year``. We will add a new technology called ``biomass_CHP`` that consumes
``biomass`` and produces ``electricity`` and ``heat``.

1. **Add a new carrier called biomass to the dataset**.

   a. Duplicate the folder ``natural_gas`` and rename it to ``biomass`` in the folder
      ``set_carriers/``.

   b. Edit the file ``attributes.json`` in the new folder ``biomass`` to reflect the characteristics of biomass.
      The changed parameters are:

      .. code-block:: JSON

          {
              "carbon_intensity_carrier_import": {
                "default_value": 0,
                "unit": "kilotons/GWh"
          },
              "carbon_intensity_carrier_export": {
                "default_value": 0,
                "unit": "kilotons/GWh"
          },
              "price_import": {
                "default_value": 100.0,
                "unit": "kiloEuro/GWh"
          }
          }

      The other parameters can be left as they are. We assume that biomass has no carbon intensity but the price
      is more than 4 times higher than natural gas.

   c. We additionally want to change the ``availability_import`` (default value 0).
      Open the file ``availability_import.csv`` in the folder ``set_carriers/biomass/``.
      Remove the content and fill with the following values:

      .. code-block::

          node, availability_import
          DE,3
          CH,0.5

      This means that in every hour 3 GW of biomass can be imported to Germany and 0.5 GW to Switzerland.

2. **Add a new conversion technology called biomass_CHP to the dataset.**
