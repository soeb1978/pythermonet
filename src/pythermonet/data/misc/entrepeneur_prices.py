from dataclasses import dataclass, field

@dataclass
class EntrepeneurPrices:
    """
    Prices that needs to be set in order to calculate the hourly paid entrepeneur prices.
    """

    # Price estimates for making the boreholes
    borehole_specialist: dict = field(default_factory= lambda: {
        "drilling_supervisor": {
            "hourly_pay": 800,
            "number_of_hours_required": 120
        },
        "drilling_operator": {
            "hourly_pay": 700,
            "number_of_hours_required": 200
        },
        "geotechnical_specialist": {
            "hourly_pay": 900,
            "number_of_hours_required": 60
        },
        "pump_installation_technician": {
            "hourly_pay": 600,
            "number_of_hours_required": 120
        },
        "electrician": {
            "hourly_pay": 650,
            "number_of_hours_required": 80
        }
    })

    # Price estimate for installing horizontal pipes in the village
    entrepeneurs: dict = field(default_factory= lambda: {
        "site_manager": {
            "hourly_pay": 750,
            "number_of_hours_required": 400
        },
        "pipe_installer": {
            "hourly_pay": 550,
            "number_of_hours_required": 1800
        },
        "excavator_operator": {
            "hourly_pay": 650,
            "number_of_hours_required": 1200
        },
        "general_laborer": {
            "hourly_pay": 450,
            "number_of_hours_required": 1600
        },
        "traffic_safety_worker": {
            "hourly_pay": 400,
            "number_of_hours_required": 600
        }
    })
