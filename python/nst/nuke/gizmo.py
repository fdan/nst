import nuke
import nukescripts


def setup_nst_nodes():
    # call this from onScriptLoad in project settings

    if not nuke.GUI:
        return

    nst_nodes = []
    nst_nodes += [x for x in nuke.allNodes() if x.Class() == 'nst']

    # debug option - include group nodes not yet saved as gizmos
    nst_nodes += [x for x in nuke.allNodes() if x.Class() == 'Group' and 'nst' in x.name()]

    for node in nst_nodes:
        setup(node)


def setup(node):
    c = [x for x in node.nodes() if x.Class() == "MLClient"]
    assert len(c) == 1
    mlc = c[0]
    mlc.showControlPanel()
    mlc.knob('connect').execute()
    create_expressions(mlc)
    node.showControlPanel()
    # nukescripts.utils.executeDeferred(create_expressions, args=(mlc))


def create_expressions(mlc):
    mlc.knob('models').setValue('Neural Style Transfer')

    # common
    mlc.knob('style_layers').setValue('[python {nuke.thisNode().parent()[\'style_layers\'].getValue()}]')
    mlc.knob('style_mips').setExpression('[python {nuke.thisNode().parent()[\'style_mips\'].getValue()}]')
    mlc.knob('style_layer_weights').setValue('[python {nuke.thisNode().parent()[\'style_layer_weights\'].getValue()}]')
    mlc.knob('style_mip_weights').setValue('[python {nuke.thisNode().parent()[\'style_mip_weights\'].getValue()}]')
    mlc.knob('content_mips').setExpression('[python {nuke.thisNode().parent()[\'content_mips\'].getValue()}]')
    mlc.knob('content_layer').setValue('[python {nuke.thisNode().parent()[\'content_layer\'].getValue()}]')
    mlc.knob('content_layer_weight').setValue('[python {nuke.thisNode().parent()[\'content_layer_weight\'].getValue()}]')
    mlc.knob('learning_rate').setExpression('[python {nuke.thisNode().parent()[\'learning_rate\'].getValue()}]')
    mlc.knob('log_iterations').setExpression('[python {int(nuke.thisNode().parent()[\'log_epochs\'].getValue())}]')
    mlc.knob('enable_update').setExpression('[python {int(nuke.thisNode().parent()[\'allow_update\'].getValue())}]')
    mlc.knob('pyramid_scale_factor').setExpression('[python {nuke.thisNode().parent()[\'mip_scale_factor\'].getValue()}]')

    # local
    mlc.knob('iterations').setExpression('[python {nuke.thisNode().parent()[\'epochs_l\'].getValue()}]')
    mlc.knob('scale').setExpression('[python {nuke.thisNode().parent()[\'scale_l\'].getValue()}]')
    mlc.knob('engine').setValue('[python {nuke.thisNode().parent()[\'engine_l\'].value().lower()}]')
    mlc.knob('optimiser').setValue('[python {nuke.thisNode().parent()[\'optimiser_l\'].value().lower()}]')

    # farm
    mlc.knob('farm_engine').setValue('[python {nuke.thisNode().parent()[\'engine_f\'].value().lower()}]')
    mlc.knob('farm_optimiser').setValue('[python {nuke.thisNode().parent()[\'optimiser_f\'].value().lower()}]')
    mlc.knob('farm_learning_rate').setValue('[python {nuke.thisNode().parent()[\'learning_rate_f\'].value()}]')
    mlc.knob('farm_scale').setExpression('[python {nuke.thisNode().parent()[\'scale_f\'].getValue()}]')
    mlc.knob('farm_iterations').setExpression('[python {nuke.thisNode().parent()[\'epochs_f\'].getValue()}]')

    mlc.knob('content_fp').setValue('[python {nuke.thisNode().input(0).knob(\'file\').value()}]')
    mlc.knob('opt_fp').setValue('[python {nuke.thisNode().input(1).knob(\'file\').value()}]')
    mlc.knob('out_fp').setValue('[python {nuke.thisNode().parent().dependent()[0].knob(\'file\').value()}]')
    mlc.knob('style1_fp').setValue('[python {nuke.thisNode().input(2).knob(\'file\').value()}]')
    mlc.knob('style1_target_fp').setValue('[python {nuke.thisNode().input(3).knob(\'file\').value()}]')
    mlc.knob('style2_fp').setValue('[python {nuke.thisNode().input(4).knob(\'file\').value()}]')
    mlc.knob('style2_target_fp').setValue('[python {nuke.thisNode().input(5).knob(\'file\').value()}]')
    mlc.knob('style3_fp').setValue('[python {nuke.thisNode().input(6).knob(\'file\').value()}]')
    mlc.knob('style3_target_fp').setValue('[python {nuke.thisNode().input(7).knob(\'file\').value()}]')

