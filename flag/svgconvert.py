#!/usr/bin/env python2

import cairo
import rsvg
from xml.dom import minidom


def convert_svg_to_png(svg_file, output_file):
    # Get the svg files content
    with open(svg_file) as f:
        svg_data = f.read()

    # Get the width / height inside of the SVG
    doc = minidom.parse(svg_file)
    width = int([path.getAttribute('width') for path
                 in doc.getElementsByTagName('svg')][0])
    height = int([path.getAttribute('height') for path
                  in doc.getElementsByTagName('svg')][0])
    doc.unlink()

    # create the png
    img = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(img)
    handler = rsvg.Handle(None, str(svg_data))
    handler.render_cairo(ctx)
    img.write_to_png(output_file)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("-f", "--file", dest="svg_file",
                        help="SVG input file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output", default="svg.png",
                        help="PNG output file", metavar="FILE")
    args = parser.parse_args()

    convert_svg_to_png(args.svg_file, args.output)