#! /usr/bin/python

import math
from pxr import Usd, UsdGeom, Gf

source_stage = Usd.Stage.Open("/scratch/dan/stepped_anim/cache.usd")
source_stage.Reload()

STEP_AMT = 2
DEBUG = True

def log(msg):
  global DEBUG
  if DEBUG:
    print(str(msg))

for prim in source_stage.TraverseAll():
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
        if ts % STEP_AMT != 1:
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
      if ts % STEP_AMT != 1:
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
      if ts % STEP_AMT != 1:
        c_points_attr.ClearAtTime(i)
        c_points_attr.Set(last_ts, time=ts)
      last_ts = c_points_attr.Get(ts)

    # Might want to do this on normals, other data
    # e.g. for pv in mesh.GetPrimvars(): etc

# Optional, not respected in Houdini, linear still used
source_stage.SetInterpolationType(Usd.InterpolationTypeHeld )

# Export new stage
#print(source_stage.ExportToString())
source_stage.Export("/scratch/dan/stepped_anim/cache_stepped.usd")