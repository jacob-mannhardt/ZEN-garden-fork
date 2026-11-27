import warnings

from zen_garden.model.component_types.parameter import GenericParameter


class MarketShareUnbounded(GenericParameter):
    """Unbounded market share of each technology."""

    name = "market_share_unbounded"
    indices = ("set_technologies",)
    doc = "Unbounded market share"
    unit_category = {}

    @classmethod
    def store_input_data(cls, element):
        """Load the technology-specific unbounded market share.

        For backward compatibility, a technology without its own
        ``market_share_unbounded`` attribute falls back to the single value
        defined in the energy system's attributes file (deprecated, see
        :meth:`after_store_input_data`).
        """
        if cls._is_defined_for(element):
            super().store_input_data(element)
            return

        energy_system = element.model_schema.energy_system
        if not cls._is_defined_for(energy_system):
            raise AttributeError(
                f"Attribute {cls.name} does not exist in input data of "
                f"{element.name} and no fallback value is defined in the energy "
                "system."
            )
        value = energy_system.element_data_loader.extract_input_data(
            cls.name,
            index_sets=cls._input_indices(element),
            unit_category=cls.unit_category,
        )
        if cls.name in energy_system.units:
            element.units[cls.name] = energy_system.units[cls.name]
        cls._store_value(element, cls.name, value)

    @classmethod
    def after_store_input_data(cls, elements):
        """Emit a single deprecation warning for all technologies that fell back
        to the energy-system value.
        """
        fallback = [
            element.name for element in elements if not cls._is_defined_for(element)
        ]
        if not fallback:
            return
        warnings.warn(
            f"{cls.name} is not defined for the technologies "
            f"{', '.join(fallback)}; using the value from the energy system "
            f"instead. Defining {cls.name} in the energy system is deprecated, "
            "define it for each technology instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    @classmethod
    def _is_defined_for(cls, element) -> bool:
        """Check whether the element's (scenario) attributes define this parameter."""
        attribute_dict, _ = element.element_data_loader.get_attribute_dict(cls.name)
        return cls.name in attribute_dict
