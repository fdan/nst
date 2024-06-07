import os
import uuid
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
        # "enable_update",
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
        "histogram_bins",
        "tv_weight",
        "laplacian_weight",
        "temporal_weight"
    ]

    bool_knobs = [
        "allowUpdate"
        # "liveRender"
    ]

    str_knobs = [
        "optimiser",
        "engine",
        "style_mip_weights",
        "style_layers",
        "mask_layers",
        "style_layer_weights",
        "content_layer",
        "random_rotate",
        "random_crop"
    ]

    enum_knobs = [
        "histogram_loss_type",
        "gram_loss_type",
        "content_loss_type",
        "laplacian_loss_type",
        "random_rotate_mode"
    ]

    # m = "[python \{nuke.thisNode().parent()\['iterations'].getValue()"

    for float_knob in float_knobs:
        try:
            mlc.knob(float_knob).setExpression('[python {nuke.thisNode().parent()[\'%s\'].getValue()}]' % float_knob)
        except:
            print('cannot set expression on:', float_knob)

    for str_knob in str_knobs:
        mlc.knob(str_knob).setValue('nuke.thisNode().parent()[\'%s\'].getValue()' % str_knob)

    for enum_knob in enum_knobs:
        mlc.knob(enum_knob).setValue('nuke.thisNode().parent()[\'%s\'].value()' % enum_knob)

    for bool_knob in bool_knobs:
        mlc.knob(bool_knob).setExpression('[python {nuke.thisNode().parent()[\'%s\'].getValue()}]' % bool_knob)

    mlc.knob("liveRender").setExpression('[python {nuke.thisNode().parent()[\'render_mode\'].getValue()}]')


def snapshot(filepath='', filename=''):
    n = nuke.thisNode()
    snaps = [x for x in n.nodes() if x.name() == "snapshot"]
    assert len(snaps) == 1
    snap = snaps[0]
    fpk = snap.knob('file')

    if not filename:
        filename = str(uuid.uuid4()).split('-')[-1]
        print('snapshot uuid:', filename)

    snapshot_dir = filepath or os.getenv('NST_SNAPSHOT_DIR')
    os.makedirs(snapshot_dir, exist_ok=True)
    ofp = '%s/%s.exr' % (snapshot_dir, filename)
    fpk.setValue(ofp)
    frame = nuke.frame()

    mmd = n.node('ModifySnapMetaData')
    mmd['metadata'].fromScript("")
    a = str(n).split('\n')

    metaKeys = []
    for i in a:
        try:
            key, value = i.split(' ')
            metaKeys.append('{set %s %s}' % (key, value))
        except:
            pass

    mmd['metadata'].fromScript("\n".join(metaKeys))

    nuke.render(snap, frame, frame)

    n.end()

    r1 = nuke.createNode('Read')
    r1.knob('file').setValue(ofp)