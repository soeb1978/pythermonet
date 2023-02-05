import os
from pathlib import Path
from thermonet.dimensioning.dimensioning import run_dimensioning
from thermonet.dimensioning.dimensioning_classes import BTESConfiguation, HorizontalConfiguration, DimensioningConfiguration, \
    DimensioningResults, PipeResult, EnergyProductionResult, HEResult, HEType

from numerous.html_report_generator import Report, Section, Div
from json2html import json2html

def make_dimensioning_report(c: DimensioningConfiguration, filename: str):

    # Run the dimensioning
    result = run_dimensioning(c)

    # Define output folder and html file
    folder = Path('./output')

    file = folder.joinpath(filename + '.html')

    # Remove previous report file if exists
    if os.path.exists(file):
        os.remove(file)

    # Create report
    report = Report(target_folder=folder, filename=filename)

    # Add info for the header and title page
    report.add_header_info(header='Thermonet',
                           title='Dimensioning Report',
                           sub_title='Thermonet',
                           sub_sub_title='',
                           footer_title='Thermonet',
                           footer_content='Dimensioning report for thermonet.'
                           )

    # Create a section for listing inputs
    section_inputs = Section(section_title="Inputs")

    # Put the inputs in a div element
    div_inputs = Div(html=json2html.convert(c.to_dict()))

    # Add the div to the section
    section_inputs.add_content({'div_inputs': div_inputs})


    # Create a section for results
    section_results = Section(section_title="Results")

    # Add the results to a div element
    div_results = Div(html=json2html.convert(result.to_dict()))

    # Add the div to the section
    section_results.add_content({'div_results': div_results})

    # Add the sections to the report
    report.add_blocks({
        'section_inputs': section_inputs,

        'section_results': section_results
    })

    # Save the report - creates the html output file
    report.save()