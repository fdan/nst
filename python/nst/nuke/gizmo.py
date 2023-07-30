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
    c = [x for x in node.nodes() if x.Class() == "MLClientLive"]
    assert len(c) == 1
    mlc = c[0]
    mlc.showControlPanel()
    mlc.knob('connect').execute()
    create_expressions(mlc, node) #defer?
    node.showControlPanel()
    # nukescripts.utils.executeDeferred(create_expressions, args=(mlc))


def create_expressions(mlc, parent):
    mlc.knob('models').setValue('Neural Style Transfer')

    float_knobs = [
        "enable_update",
        "iterations",
        "log_iterations",
        "batch_size",
        "learning_rate",
        "style_mips",
        "style_pyramid_span",
        "style_zoom",
        "content_layer_weight",
        "gram_weight",
        "histogram_weight",
        "tv_weight",
        "laplacian_weight",
    ]

    str_knobs = [
        "optimiser",
        "engine",
        "style_mip_weights",
        "style_layers",
        "mask_layers",
        "style_layer_weights",
        "content_layer",
    ]

    for float_knob in float_knobs:
        try:
            mlc.knob(float_knob).setExpression('[python {nuke.thisNode().parent()[\'%s\'].getValue()}]' % float_knob)
            # mlc.knob(float_knob).setExpression('%s.%s' % (parent.name(), float_knob))
        except:
            print(1.1, float_knob)

    for str_knob in str_knobs:
        print(1.2, str_knob)
        mlc.knob(str_knob).setValue('nuke.thisNode().parent()[\'%s\'].getValue()' % str_knob)
        # value = str(parent[str_knob].getValue())
        # mlc.knob(str_knob).setValue(value)

