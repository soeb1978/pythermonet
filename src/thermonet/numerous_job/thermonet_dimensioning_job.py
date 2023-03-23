from numerous.image_tools.job import NumerousBaseJob, ExitCode
from numerous.image_tools.app import run_job
import logging
import os
from ..reporting.report import make_dimensioning_report
from ..dimensioning.thermonet_classes import HHEconfig, BHEconfig, Brine, Heatpump, Thermonet, FullDimension
from pathlib import Path

LOG_LEVEL = os.getenv('LOG_LEVEL', logging.DEBUG)

class ThermonetJob(NumerousBaseJob):

    def __init__(self):
        super(ThermonetJob, self).__init__()

        self.report = None
        self.template = f"{os.path.dirname(__file__)}/report/template/report_template_em.html"
        self.logger = logging.getLogger('numerous-report-job')
        self.logger.setLevel(level=LOG_LEVEL)

    def run_job(self) -> ExitCode:
        self.logger.info("adding report content")
        self.run_dimensioning_report()

        self.logger.warning('job terminated')

        return ExitCode.COMPLETED

    def run_dimensioning_report(self):
        report_file = "report"

        thermonet = Thermonet(**self.system.components["thermonet"].constants)
        heatpump = Heatpump(**self.system.components["heatpump"].constants)
        brine = Brine(**self.system.components["brine"].constants)
        if "borehole_heat_exchanger" in self.system.components:
            heat_exchanger = BHEconfig(**self.system.components["borehole_heat_exchanger"].constants)
        else:
            heat_exchanger = HHEconfig(**self.system.components["horizontal_heat_exchanger"].constants)

        if "prosumers_file" in self.system.components:
            prosumer_file = Path(self.system.components["prosumers_file"].constants["prosumer_file"])
        else:
            raise NotImplementedError("The prosumer list is not implemented")

        if "sections_file" in self.system.components:
            topology_file = Path(self.system.components["sections_file"].constants["topology_file"])
        else:
            raise NotImplementedError("The sections list is not implemented")

        configuration = FullDimension(
            thermonet=thermonet,
            heatpump=heatpump,
            brine=brine,
            source_config=heat_exchanger,
            PID="Test",
            HPFN=prosumer_file,  # Input file containing heat pump information
            TOPOFN=topology_file,
            lp=0.4

        )

        make_dimensioning_report(configuration, filename=report_file)
        self.app.client.upload_file("output/"+report_file+".html", "report")


def run_thermonet_job():
    run_job(numerous_job=ThermonetJob(), appname="thermonet-dimensioning", model_folder=".")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_thermonet_job()
