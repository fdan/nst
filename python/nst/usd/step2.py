#! /usr/bin/python
from optparse import OptionParser
import math

from pxr import Usd, UsdGeom, Gf


def log(msg):
  global DEBUG
  if DEBUG:
    print(str(msg))


def main():
  p = OptionParser()
  p.add_option("-i", "--in", dest='in_usd', action='store')
  p.add_option("-o", "--out", dest='out_usd', action='store')
  p.add_option("-s", "--step", dest='step', action='store')
  p.add_option("-d", "--debug", dest='debug', action='store_true')
  opts, args = p.parse_args()

  stage = Usd.Stage.Open(opts.in_usd)
  stage.Reload()

  global DEBUG
  if opts.debug:
    DEBUG = True
  else:
    DEBUG = False

  step = opts.step or 2

  stage = _step_anim(stage, step)
  #print(source_stage.ExportToString())
  stage.Export(opts.out_usd)


def _step_anim(stage, step):

  for prim in stage.TraverseAll():
    log(prim.GetName())

    # - Clear xform time samples
    if prim.IsA(UsdGeom.Xformable):
      xform = UsdGeom.Xform(prim)
      # For each operator (xform ops can be stacked)
      for op in xform.GetOrderedXformOps():
        op_attr = op.GetAttr()
        log(op_attr.GetTimeSamples())
        # Clear moduloed time samples
        last_ts = None
        for i in op_attr.GetTimeSamples():
          ts = math.floor(i)
          if ts % step != 1:
            op_attr.ClearAtTime(i)
            op_attr.Set(last_ts, time=ts)
          last_ts = op_attr.Get(ts)

    # - Clear mesh time samples
    if prim.IsA(UsdGeom.Mesh):
      mesh = UsdGeom.Mesh(prim)
      points_attr = mesh.GetPointsAttr()
      # Clear moduloed time samples
      last_ts = None
      for i in points_attr.GetTimeSamples():
        ts = math.floor(i)
        if ts % step != 1:
          points_attr.ClearAtTime(i)
          points_attr.Set(last_ts, time=ts)
        last_ts = points_attr.Get(ts)

    # - Clear curve time samples
    if prim.IsA(UsdGeom.BasisCurves):
      curves = UsdGeom.BasisCurves(prim)
      c_points_attr = curves.GetPointsAttr()
      # Clear moduloed time samples
      last_ts = None
      for i in c_points_attr.GetTimeSamples():
        ts = math.floor(i)
        if ts % step != 1:
          c_points_attr.ClearAtTime(i)
          c_points_attr.Set(last_ts, time=ts)
        last_ts = c_points_attr.Get(ts)

    # Might want to do this on normals, other data
    # e.g. for pv in mesh.GetPrimvars(): etc

  # Optional, not respected in Houdini, linear still used
  stage.SetInterpolationType(Usd.InterpolationTypeHeld )
  return stage


if "__main__" == __name__:
    main()

