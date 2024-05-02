// Copyright (c) 2018 Foundry.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*************************************************************************


// https://learn.foundry.com/nuke/developers/130/ndkreference/Plugins/classDD_1_1Image_1_1Op.html#ad19589f141ad102c40fab9c26a87c5c8
// https://learn.foundry.com/nuke/developers/130/ndkreference/Plugins/classDD_1_1Image_1_1Op.html#a009bdb3fc6cf7b4f33134c71585f2f4a
// void DD::Image::Op::cached

#include <cstring>
#include <iostream>
#include <thread>
#include <chrono>

#include "MLClientLive.h"

const char* const MLClientLive::kClassName = "MLClientLive";
const char* const MLClientLive::kHelpString =
  "Connects to a Python server for Machine Learning inference.";

const char* const MLClientLive::kDefaultHostName = "172.22.15.27";
const int         MLClientLive::kDefaultPortNumber = 55555;

const DD::Image::ChannelSet MLClientLive::kDefaultChannels = DD::Image::Mask_RGB;
const int MLClientLive::kDefaultNumberOfChannels = MLClientLive::kDefaultChannels.size();

using namespace DD::Image;

static void listener(unsigned, unsigned, void* d);

/*! This is a function that creates an instance of the operator, and is
   needed for the Iop::Desc
 */
static Iop* MLClientLiveCreate(Node* node)
{
  return new MLClientLive(node);
}

/*! The Iop::Description is how NUKE knows what the name of the operator is,
   how to create one, and the menu item to show the user. The menu item may be
   0 if you do not want the operator to be visible.
 */
const Iop::Description MLClientLive::description(MLClientLive::kClassName, 0, MLClientLiveCreate);

//! Constructor. Initialize user controls to their default values.
MLClientLive::MLClientLive(Node* node)
: DD::Image::PlanarIop(node)
, _host(MLClientLive::kDefaultHostName)
, _hostIsValid(true)
, _port(MLClientLive::kDefaultPortNumber)
, _portIsValid(true)
, _chosenModel(0)
, _modelSelected(false)
, _showDynamic(false)
, _numNewKnobs(0)
, _modelManager(this)
{
    killthread = false;
    terminate = false;
    allowUpdate = true;
    responseStatus = CanSend;
    jobStatus = Completed;
    batch = 0;

    ThreadId this_id = Thread::GetThreadId();
    std::stringstream  ss1;
    ss1 << "[MLCL] constructor called, thread id: " << this_id;
    MLClientComms::DebugPrint(ss1.str());
}

MLClientLive::~MLClientLive() {
    MLClientComms::DebugPrint("MLClientLive destructor called");
}

void MLClientLive::append(Hash& hash){
    hash.append(hashCtr);
//    hash.append(batch);
}

//! The maximum number of input connections the operator can have.
int MLClientLive::maximum_inputs() const
{
  if (haveValidModelInfo() && _modelSelected) {
    return _numInputs[_chosenModel];
  }
  else {
    return 1;
  }
}

//! The minimum number of input connections the operator can have.
int MLClientLive::minimum_inputs() const
{ 
  if (haveValidModelInfo() && _modelSelected) {
    return _numInputs[_chosenModel];
  }
  else {
    return 1;
  }
}

/*! Return the text Nuke should draw on the arrow head for input \a input
    in the DAG window. This should be a very short string, one letter
    ideally. Return null or an empty string to not label the arrow.
*/
const char* MLClientLive::input_label(int input, char* buffer) const
{
  if (!haveValidModelInfo() || !_modelSelected) {
    return "";
  }
  else {
    if ((input < _inputNames[_chosenModel].size()) && (_chosenModel < _inputNames.size())) {
      return _inputNames[_chosenModel][input].c_str();
    }
    else {
      return "";
    }
  }
}

bool MLClientLive::useStripes() const
{
  return false;
}

bool MLClientLive::renderFullPlanes() const
{
  return true;
}

MLClientModelManager& MLClientLive::getModelManager()
{
  return dynamic_cast<MLClientLive*>(firstOp())->_modelManager;
}

int MLClientLive::getNumNewKnobs()
{
  return dynamic_cast<MLClientLive*>(firstOp())->_numNewKnobs;
}

void MLClientLive::setNumNewKnobs(int i)
{
  dynamic_cast<MLClientLive*>(firstOp())->_numNewKnobs = i;
}

void MLClientLive::_validate(bool forReal)
{
  // Try connect to the server, erroring if it can't connect.
  std::string connectErrorMsg;
  if (!haveValidModelInfo() && !refreshModelsAndKnobsFromServer(connectErrorMsg)) {
    error(connectErrorMsg.c_str());
  }
  // The only other thing needed to do in validate is copy the image info.
  copy_info();
}

void MLClientLive::getRequests(const Box& box, const ChannelSet& channels, int count,
                           RequestOutput &reqData) const
{
  // request all input input as we are going to search the whole input area
  for (int i = 0, endI = getInputs().size(); i < endI; i++) {
    const ChannelSet readChannels = input(i)->info().channels();
    input(i)->request(readChannels, count);
  }
}


void MLClientLive::renderStripe(ImagePlane& imagePlane) {
    ThreadId this_id = Thread::GetThreadId();
    std::stringstream ss0;
    ss0 << "\n[MLCL] renderStripe entered, (thread id: " << this_id << ")";
    MLClientComms::DebugPrint(ss0.str());

    MLClientComms::DebugPrint("[MLCL] renderStripe entered");

    DD::Image::Knob *itKnob = knob("iterations");
    DD::Image::Knob *bsKnob = knob("batch_size");
    DD::Image::Knob *liveRenderKnob = knob("liveRender");
    int iterations = itKnob->get_value();
    int batch_size = bsKnob->get_value();

    std::stringstream ss00;
    ss00 << "renderStripe iterations: " << iterations;
    MLClientComms::DebugPrint(ss00.str());

//    bool liveRender = liveRenderKnob->get_value();

//    std::stringstream ss00;
//    ss00 << "liveRender value" << liveRender << " ";
//    MLClientComms::Print(ss00.str());

    // why does this not work here?  b/c renderStripe is threaded?
//  auto logKnob = knob("Log");
//  std::string msg = "renderStripe entered";
//  logKnob->set_text(msg.c_str());

    if (iterations == 0) {
        MLClientComms::DebugPrint("[MLCL] renderStripe saw iterations size 0, returning");
        return;
    }

    if (batch_size == 0) {
        MLClientComms::DebugPrint("[MLCL] renderStripe saw batch_size 0, returning");
        return;
    }

    DD::Image::Knob *allowUpdateKnob = knob("allowUpdate");
    if (!allowUpdateKnob->get_value()) {
        MLClientComms::DebugPrint("[MLCL] renderStripe saw allowUpdate==False, returning");
        // this should allow the incoming plane to pass through.  ideally would instead
        // pass through the cached image, but not sure how
        input0().fetchPlane(imagePlane);
//      input0().fetchPlane(getCache());
//      getCache();
        return;
    }

    if (aborted()) {
        MLClientComms::DebugPrint("[MLCL] renderStripe aborted, returning");
        return;
    }

    if (cancelled()) {
        MLClientComms::DebugPrint("[MLCL] renderStripe cancelled, returning");
        return;
    }

    batches = iterations / batch_size;

    if (!haveValidModelInfo() || !_modelSelected) {
        input0().fetchPlane(imagePlane);
//      MLClientComms::Vprint("[MLCL] renderStripe returning");
        return;
    }

    std::string errorMsg;

    // live render mode
    if (liveRenderKnob->get_value() == 1) {
        MLClientComms::Print("Live render");

        if (responseStatus == CanSend) {

            if (jobStatus == InProgress) {
                MLClientComms::DebugPrint("[MLCL] renderStripe CanSend called when a job is in progress, returning");
                return;
            }

            std::stringstream ss1;
            ss1 << "\n[MLCL] Render job started, (thread id: " << this_id << ")";
            std::string s1 = ss1.str();
            MLClientComms::Print(s1);
                MLClientComms::DebugPrint("[MLCL] renderStripe saw status CanSend");
            terminate = false;
            std::stringstream ss2;
            ss2 << "[MLCL] renderStripe spawned listener";
            MLClientComms::DebugPrint(ss2.str());
            input0().fetchPlane(imagePlane);
            jobStatus = InProgress;
            killthread = false;
            Thread::spawn(::listener, 1, this);
        } else if (responseStatus == HasReceived) {
            std::stringstream ss5;
            ss5 << "[MLCL] renderStripe saw status: HasReceived batch " << batch - 1 << ", processing result (thread id: "
                << this_id << ")";
            MLClientComms::DebugPrint(ss5.str());

            std::stringstream ss6;
            ss6 << "[MLCL] Received batch " << batch - 1;
            std::string s6 = ss6.str();
            MLClientComms::Print(s6);

            if (!fillImagePlane(imagePlane, errorMsg)) {
                MLClientComms::Print(errorMsg);
                error(errorMsg.c_str());
            }

            if (jobStatus == Completed) {
                responseStatus = CanSend;
    //          killthread = false;
    //          terminate = false;
                std::string s7 = "Job completed\n";
                MLClientComms::Print(s7);
            }
        }
    }

    // do a disk render
    else{
        MLClientComms::Print("Disk render");
        MLClientComms comms(_host, _port);
        std::string errorMsg;
        mlserver::RequestInference* requestInference = new mlserver::RequestInference;

        bool clearCache = true;
        int _batch = 1;
        int _batches = 1;

        getInferenceRequest(_host, _port, errorMsg, requestInference, clearCache, _batches, _batch, iterations);
        comms.sendInferenceRequest(*requestInference);
        comms.readInferenceResponse(responseWrapper);

        if (!fillImagePlane(imagePlane, errorMsg)) {
            MLClientComms::Print(errorMsg);
            error(errorMsg.c_str());
        }
    }

    MLClientComms::DebugPrint("[MLCL] renderStripe returning");
    return;
}

    // if we reached this point, reset everything for next call
//    responseStatus = CanSend;
//    killthread = false;
//    terminate = false;
//    batch = 0;
//    MLClientComms::Vprint("[MLCL] renderStripe returning");
//    return;
//  }

  // Check again if we hit abort during processing
//  if (aborted() || cancelled()) {
//    MLClientComms::Vprint("Aborted without processing image.");
//    return;
//  }

  // If we reached here by default let's pull an image from input0() so
  // that it's at least passing something through.
//  input0().fetchPlane(imagePlane);
//}

bool MLClientLive::refreshModelsAndKnobsFromServer(std::string& errorMsg)
{
  // Before trying to connect, ensure ports and hostname are valid.
  if (!_portIsValid) {
    errorMsg = "Port is invalid.";
    return false;
  }
  if(!_hostIsValid) {
    errorMsg = "Hostname is invalid.";
    return false;
  }

  // pull model info
  {
    MLClientComms comms(_host, _port);

    if (!comms.isConnected()) {
      errorMsg = "Could not connect to server. Please check your host / port numbers.";
      return false;
    }

    // Try pull the model info into the responseWrapper
    if(!comms.sendInfoRequestAndReadInfoResponse(responseWrapper, errorMsg)) {
      return false;
    }
  }
  // Parse message and fill in menu items for enumeration knob
  _serverModels.clear();
  _numInputs.clear();
  _inputNames.clear();
  std::vector<std::string> modelNames;
  int numModels = responseWrapper.r1().num_models();
  std::stringstream ss;
  ss << "[MLCL] Server can serve " << std::to_string(numModels) << " models" << std::endl;
  ss << "-----------------------------------------------";
  MLClientComms::Print(ss.str());
  for (int i = 0; i < numModels; i++) {
    mlserver::Model model;
    model = responseWrapper.r1().models(i);
    modelNames.push_back(model.label());
    _serverModels.push_back(model);
    _numInputs.push_back(model.inputs_size());
    std::vector<std::string> names;
    for (int j = 0; j < model.inputs_size(); j++) {
      mlserver::ImagePrototype p;
      p = model.inputs(j);
      names.push_back(p.name());
    }
    _inputNames.push_back(names);
  }

  // Sanity check that some models were returned
  if (_serverModels.size() == 0) {
    errorMsg = "Server returned no models.";
    return false;
  }

  // Change enumeration knob choices
  Enumeration_KnobI* pSelectModelEnum = _selectedModelknob->enumerationKnob();
  pSelectModelEnum->menu(modelNames);

  if (_chosenModel >= (int)numModels) {
    _selectedModelknob->set_value(0);
    _chosenModel = 0;
  }

  // We try to select the model saved in the serial serialiseKnob if any.
  bool restoreModel = false;
  MLClientModelKnob* modelKnob = nullptr;
  DD::Image::Knob* k = knob("serialiseKnob");
  if(k != nullptr) {
    modelKnob = dynamic_cast<MLClientModelKnob*>(k);
    if(modelKnob != nullptr) {
      std::string modelLabel = modelKnob->getModel();
      if(modelLabel != "") {
        const std::vector<std::string>& models = pSelectModelEnum->menu();
        int i = 0;
        for(auto& m : models) {
          if(m == modelLabel) {
            _chosenModel = i;
            _selectedModelknob->set_value(_chosenModel);
            restoreModel = true;
            break;
          }
          i++;
        }
      }      
    }
  }

  // Set member variables to indicate our connections and model set-up succeeded.
  _modelSelected = true;
  _showDynamic = true;

  // Update the dynamic knobs
  const mlserver::Model m = _serverModels[_chosenModel];
  if (this == this->firstOp()) {
    getModelManager().parseOptions(m);
  }
  setNumNewKnobs(replace_knobs(knob("models"), getNumNewKnobs(), addDynamicKnobs, this->firstOp()));

  // If we have restored a model, we also need to restore its parameters
  // now that its knobs have been created,
  if(restoreModel && (modelKnob != nullptr)) {
    for(const std::pair<std::string, std::string>& keyVal: modelKnob->getParameters()) {
      restoreKnobValue(keyVal.first, keyVal.second);
    }
  }

  // Return true if control made it here, success.
  return true;
}

void MLClientLive::restoreKnobValue(const std::string& knobName, const std::string& value)
{
  // We look for the corresponding knob 
  DD::Image::Knob* paramKnob = knob(knobName.c_str());
  if(paramKnob != nullptr) {
    // Is this an animation curve?
    if(value.substr(0, 6) == "{curve") {
      // This is a curve, we remove the { }
      std::string curveString = value.substr(1, value.find("}") - 1);
      paramKnob->set_animation(curveString.c_str(), 0);
    } 
    else if(value.substr(0, 1) == "{") {
      // That's an expression
      std::string expressionString = value.substr(1, value.find("}") - 1);
      // If the expression is within double quote, we need to extract it
      if(expressionString.substr(0, 1) == "\"") {
        expressionString.erase(0, 1);
        expressionString = expressionString.substr(0, expressionString.find("\""));
      } else {
        // The expression might be followed by keys that we ignore here.
        if(expressionString.find(" ") != std::string::npos) {
          expressionString = expressionString.substr(0, expressionString.find(" "));
        }
      }
      paramKnob->set_expression(expressionString.c_str(), 0);
    }
    else {
      // That's one value
      paramKnob->set_text(value.c_str());
    }
  }
}

//! Return whether we successfully managed to pull model
//! info from the server at some time in the past, and the selected model is
//! valid.
bool MLClientLive::haveValidModelInfo() const
{
  return _serverModels.size() > 0 && _serverModels.size() > _chosenModel;
}

bool MLClientLive::getInferenceRequest(const std::string& hostStr,
                                       int port,
                                       std::string& errorMsg,
                                       mlserver::RequestInference* requestInference,
                                       bool clearCache,
                                       int batch_total,
                                       int batch_current,
                                       int iterations){

    ThreadId this_id = Thread::GetThreadId();
    std::stringstream  ss1;
    ss1 << "[MLCL] getInferenceRequest entered, thread id: " << this_id;
    MLClientComms::DebugPrint(ss1.str());

    const Box imageFormat = info().format();

    mlserver::Model* m = new mlserver::Model(_serverModels[_chosenModel]);
    getModelManager().updateOptions(*m);
    requestInference->set_allocated_model(m);

    if (clearCache){
        requestInference->set_clearcache(1);
    }
    else{
        requestInference->set_clearcache(0);
    }

    requestInference->set_batch_total(batch_total);
    requestInference->set_batch_current(batch_current);
    requestInference->set_iterations(iterations);

    // todo: the opt image is the first input.  get from cached value
    for (int i = 0; i < node_inputs(); i++) {

        const ChannelSet readChannels = input(i)->info().channels();

        // Create an ImagePlane, and read each input into it.
        // Get our input & sanity check
        DD::Image::Iop *inputIop = dynamic_cast<DD::Image::Iop *>(input(i));
        if (inputIop == NULL) {
            errorMsg = "[MLCL] Input is empty or not connected.";
            return false;
        }

        // Checking before validating inputs
        if (aborted()) {
            errorMsg = "[MLCL] Process aborted before validating inputs.";
            return false;
        }

        // Try validate & request the input, this should be quick if the data
        // has already been requested.
        if (!inputIop->tryValidate(/*force*/true)) {
            errorMsg = "[MLCL] Unable to validate input.";
            return false;
        }

        Box imageBounds = inputIop->info();
        const int fx = imageBounds.x();
        const int fy = imageBounds.y();
        const int fr = imageBounds.r();
        const int ft = imageBounds.t();

        // Request our default channels, for our own bounding box
        inputIop->request(fx, fy, fr, ft, readChannels, 0);

        ImagePlane plane(imageBounds, /*packed*/ true, readChannels, readChannels.size());

//        // if the response wrapper contains a result, and i is the opt image,
//        // prefill using the result of the last batch:
//        if (responseWrapper.has_r2() && i ==1){
//            MLClientComms::Vprint("1");
//            fillImagePlane(plane, errorMsg);
//        }

        inputIop->fetchPlane(plane);

        // Sanity check that that the plane was filled successfully, and nothing
        // was interrupted.
        if (plane.usage() == 0) {
            errorMsg = "[MLCL] No image data fetched from input.";
            return false;
        }

        // Checking after fetching inputs
        if (aborted()) {
            errorMsg = "[MLCL] Process aborted after fetching inputs.";
            return false;
        }

        // Set up our message
        mlserver::Image *image = requestInference->add_images();
        image->set_width(fr);
        image->set_height(ft);
        image->set_channels(readChannels.size());

        // Set up our temp contiguous buffer
        size_t byteBufferSize = fr * ft * readChannels.size() * sizeof(float);
        if (byteBufferSize == 0) {
            errorMsg = "[MLCL] Image size is zero.";
            return false;
        }
        // Create and zero our buffer
        byte *byteBuffer = new byte[byteBufferSize];
        std::memset(byteBuffer, 0, byteBufferSize);

        float *floatBuffer = (float *) byteBuffer;
        for (int z = 0; z < readChannels.size(); z++) {

            const int chanStride = z * fr * ft;

            for (int ry = fy; ry < ft; ry++) {
                const int rowStride = ry * fr;
                ImageTileReadOnlyPtr tile = plane.readableAt(ry, z);
                for (int rx = fx, currentPos = 0; rx < fr; rx++) {
                    size_t fullPos = chanStride + rowStride + currentPos++;
                    // todo: crashes here if bbox is out of bounds, check and avoid segfault
                    floatBuffer[fullPos] = tile[rx];
                }
            }
        }

        image->set_image(byteBuffer, byteBufferSize);
        delete[] byteBuffer;
    }
    return requestInference;
}

bool MLClientLive::fillImagePlane(DD::Image::ImagePlane& imagePlane, std::string& errorMsg)
{
  // Sanity check, make sure the response actually contains an image.
  if (!responseWrapper.has_r2() || responseWrapper.r2().num_images() == 0) {
    errorMsg = "No image found in message response.";
    return false;
  }

  // Validate ourself before proceeding, this ensures if this is being invoked by a button press
  // then it's set up correctly. This will return immediately if it's already set up.
  if (!tryValidate(/*for_real*/true)) {
    errorMsg = "Could not set-up node correctly.";
    return false;
  }

  // Get the resulting image data
  const mlserver::Image &imageMessage = responseWrapper.r2().images(0);

  // Verify that the image passed back to us is of the same format as the input
  // format (note, the bounds of the imagePlane may be different, e.g. if there's
  // a Crop on the input.)
  const Box imageFormat = info().format();
  if (imageMessage.width() != imageFormat.w() || imageMessage.height() != imageFormat.h()) {
    errorMsg = "Received Image has dimensions different than expected";
    return false;
  }

  // Set the dimensions of the imagePlane, note this can be different than the format.
  // Clip it to the intersection of the image format.
  Box imageBounds = imagePlane.bounds();
  imageBounds.intersect(imageFormat);
  const int fx = imageBounds.x();
  const int fy = imageBounds.y();
  const int fr = imageBounds.r();
  const int ft = imageBounds.t();

  // This is going to copy back the minimum intersection of channels between
  // what's required to fill in imagePlane, and what's been returned
  // in the response. This allows us to gracefully handle cases where the returned
  // image has too few channels, or when the imagePlane has too many.
  const size_t numChannelsToCopy = (imageMessage.channels() < imagePlane.channels().size()) ? imageMessage.channels() : imagePlane.channels().size();

  // Copy the data
  const char* imageByteDataPtr = imageMessage.image().c_str();

  // Sanity check our image has the correct number of elements
  const size_t numImageElements = imageMessage.image().size() / sizeof(float);
  const size_t numImageElementsToCopy = numChannelsToCopy * (ft - fy) * (fr - fx);

  if (numImageElements < numImageElementsToCopy) {
    errorMsg = "Received Image has insuffient elements.";
    return false;
  }

  // Allow the imagePlane to be writable
  imagePlane.makeWritable();

  float* imageFloatDataPtr = (float*)imageByteDataPtr;
  for (int z = 0; z < numChannelsToCopy; z++) {
    const int chanStride = z * imageFormat.w() * imageFormat.h();

    for (int ry = fy; ry < ft; ry++) {
      const int rowStride = ry * imageFormat.w();

      for (int rx = fx, currentPos = 0; rx < fr; rx++) {
        int fullPos = chanStride + rowStride + currentPos++;
        imagePlane.writableAt(rx, ry, z) = imageFloatDataPtr[fullPos];
      }
    }
  }

  return true;
}

void MLClientLive::addDynamicKnobs(void* p, Knob_Callback f)
{
  if (((MLClientLive *)p)->getShowDynamic()) {
    for (int i = 0; i < ((MLClientLive *)p)->getModelManager().getNumOfInts(); i++) {
      std::string name = ((MLClientLive *)p)->getModelManager().getDynamicIntName(i);
      std::string label = ((MLClientLive *)p)->getModelManager().getDynamicIntName(i);
      Int_knob(f, ((MLClientLive *)p)->getModelManager().getDynamicIntValue(i), name.c_str(), label.c_str());
      SetFlags(f, Knob::DO_NOT_WRITE);
      Newline(f, " ");
    }
    for (int i = 0; i < ((MLClientLive *)p)->getModelManager().getNumOfFloats(); i++) {
      std::string name = ((MLClientLive *)p)->getModelManager().getDynamicFloatName(i);
      std::string label = ((MLClientLive *)p)->getModelManager().getDynamicFloatName(i);
      Float_knob(f, ((MLClientLive *)p)->getModelManager().getDynamicFloatValue(i), name.c_str(), label.c_str());
      ClearFlags(f, Knob::SLIDER);
      SetFlags(f, Knob::DO_NOT_WRITE);
      Newline(f, " ");
    }
    for (int i = 0; i < ((MLClientLive *)p)->getModelManager().getNumOfBools(); i++) {
      std::string name = ((MLClientLive *)p)->getModelManager().getDynamicBoolName(i);
      std::string label = ((MLClientLive *)p)->getModelManager().getDynamicBoolName(i);
      Bool_knob(f, ((MLClientLive *)p)->getModelManager().getDynamicBoolValue(i), name.c_str(), label.c_str());
      SetFlags(f, Knob::DO_NOT_WRITE);
      Newline(f, " ");
    }
    for (int i = 0; i < ((MLClientLive *)p)->getModelManager().getNumOfStrings(); i++) {
      std::string name = ((MLClientLive *)p)->getModelManager().getDynamicStringName(i);
      std::string label = ((MLClientLive *)p)->getModelManager().getDynamicStringName(i);
      String_knob(f, ((MLClientLive *)p)->getModelManager().getDynamicStringValue(i), name.c_str(), label.c_str());
      SetFlags(f, Knob::DO_NOT_WRITE);
      Newline(f, " ");
    }
    for (int i = 0; i < ((MLClientLive *)p)->getModelManager().getNumOfButtons(); i++) {
      std::string name = ((MLClientLive *)p)->getModelManager().getDynamicButtonName(i);
      std::string label = ((MLClientLive *)p)->getModelManager().getDynamicButtonName(i);
      Button(f, name.c_str(), label.c_str());
      Newline(f, " ");
    }
  }
}

void MLClientLive::knobs(Knob_Callback f)
{
  String_knob(f, &_host, "host");
  SetFlags(f, Knob::ALWAYS_SAVE);

  Int_knob(f, &_port, "port");
  SetFlags(f, Knob::ALWAYS_SAVE);

  Button(f, "connect", "Connect");
  Divider(f, "  ");

  Bool_knob(f, &allowUpdate, "allowUpdate");
  Divider(f, "  ");

  Bool_knob(f, &allowUpdate, "liveRender");
  Divider(f, "  ");

  Button(f, "render", "Render");
  Divider(f, "  ");

  Button(f, "cancel", "Cancel");
  Divider(f, "  ");

  static const char* static_choices[] = {0};
  Knob* knob = Enumeration_knob(f, &_chosenModel, static_choices, "models", "Models");
  if (knob) {
    _selectedModelknob = knob;
  }
  SetFlags(f, Knob::SAVE_MENU);

  // We create a knob to save/load the current state of the dynamic knobs.
  if(f.makeKnobs()) {
    CustomKnob1(MLClientModelKnob, f, this, "serialiseKnob");
  }

  if (!f.makeKnobs()) {
    MLClientLive::addDynamicKnobs(this->firstOp(), f);
  }

//  const char* _multilineStringKnob = "barfoo";
//  Multiline_String_knob(f, &_multilineStringKnob, "Log");
//  Divider(f, "  ");

}

int MLClientLive::knob_changed(Knob* knobChanged)
{
  if (knobChanged->is("cancel")){
      MLClientComms::Print("cancel knob changed");

//      DD::Image::Knob *itKnob = knob("iterations");
//      int _iterations = itKnob->get_value();
//      std::stringstream ss00;
//      ss00 << "renderStripe iterations: " << _iterations;
//      MLClientComms::Print(ss00.str());
//      ThreadId this_id = Thread::GetThreadId();
//      std::stringstream ss1;
//      ss1 << "knob_changed thead id: " << this_id;
//      MLClientComms::Print(ss1.str());
//      auto logKnob = knob("Log");
//      std::string msg = "foobar";
//      logKnob->set_text(msg.c_str());

      terminate = true;
      return 1;
  }

//  if (knobChanged->is("render")){
//      MLClientComms::Vprint("render knob changed");
//      hashCtr += 1;
//      asapUpdate();
//      return 1;
//  }

  if (knobChanged->is("host")) {
    if (!MLClientComms::ValidateHostName(_host)) {
      error("Please insert a valid host ipv4 or ipv6 address.");
      _hostIsValid = false;
    }
    else {
      _hostIsValid = true;
    }
    return 1;
  }

  if (knobChanged->is("port")) {
    if (_port > 65535 || _port < 0) {
      error("Port out of range.");
      _portIsValid = false;
    }
    else {
      _portIsValid = true;
    }
    return 1;
  }

  if (knobChanged->is("connect")) {

    // if reconneng, we need to reset the status to Ready
    responseStatus = CanSend;

    std::string connectErrorMsg;
    if (!refreshModelsAndKnobsFromServer(connectErrorMsg)) {
      error(connectErrorMsg.c_str());
    }
    return 1;
  }

  if (knobChanged->is("models")) {
    // Sanity check that some models exist
    if (haveValidModelInfo()) {
      const mlserver::Model m = _serverModels[_chosenModel];
      getModelManager().parseOptions(m);
      setNumNewKnobs(replace_knobs(knob("models"), getNumNewKnobs(), addDynamicKnobs, this->firstOp()));
    }
    return 1;
  }

  return 0;
}

//! Return the name of the class.
const char* MLClientLive::Class() const
{
  return MLClientLive::kClassName;
}

const char* MLClientLive::node_help() const
{
  return MLClientLive::kHelpString;
}

bool MLClientLive::getShowDynamic() const
{
  return _showDynamic && haveValidModelInfo();
}



static void listener(unsigned index, unsigned nThreads, void* d) {

    ThreadId this_id = Thread::GetThreadId();
    std::stringstream ss10;
    ss10 << "[LISTENER] Listener thread id: " << this_id;
    MLClientComms::DebugPrint(ss10.str());

    Guard guard(((MLClientLive *) d)->_lock);
//    MLClientComms::Vprint("[LISTENER] listener sleeping");
//    std::this_thread::sleep_for(std::chrono::seconds(5));
//    MLClientComms::Vprint("[LISTENER] listener awake");

// this doesn't work - wrong thread?
//    auto logKnob = ((MLClientLive*)d)->knob("Log");
//    std::string msg = "listener";
//    logKnob->set_text(msg.c_str());

    std::stringstream ss11;
    ss11 << "[LISTENER] total batches: " << ((MLClientLive*)d)->batches;
    MLClientComms::Print(ss11.str());

    while (!((MLClientLive *) d)->killthread) {
        MLClientComms::DebugPrint("[LISTENER] listener thread not killed");

        bool clearCache = true;

        for (int b = 0; b < ((MLClientLive*)d)->batches; b++) {
            std::stringstream ss12;
            ss12 << "[LISTENER] starting batch: " << b;
            MLClientComms::Print(ss12.str());
            bool terminate = ((MLClientLive *)d)->terminate;

            std::stringstream ss13;
            ss13 << "[LISTENER] terminate status: " << terminate;
            MLClientComms::Print(ss13.str());

            if (terminate){
                MLClientComms::Print("[LISTENER] terminating...");
                ((MLClientLive *) d)->terminate = false;
                ((MLClientLive *) d)->jobStatus = Completed;
                ((MLClientLive *) d)->killthread = true;
                break;
            }
            else{
                MLClientComms::Print("[LISTENER] not terminating...");
            }

            MLClientComms comms(((MLClientLive *) d)->_host, ((MLClientLive *) d)->_port);
            std::string errorMsgL;
            mlserver::RequestInference* requestInference = new mlserver::RequestInference;

            if (!b==0){
                clearCache = false;
            }

            // this seems to stop the segfault, and explains why it won't segfault witha breakpoint set
            // anywhere in the listener.  however it's not a solution and points to a race condition
            // with the listener's thread and the MLClientLive object's thread.
            // be interesting to see if we need to do this for each batch, or just at the start of the listener.
//            std::this_thread::sleep_for(std::chrono::seconds(10));

//            DD::Image::Knob *itKnob = ((MLClientLive*)d)->knob("iterations");
//            int iterations = itKnob->get_value();

            DD::Image::Knob *bsKnob = ((MLClientLive*)d)->knob("batch_size");
            int batch_size = bsKnob->get_value();

            ((MLClientLive*)d)->getInferenceRequest(((MLClientLive*)d)->_host,
                                                    ((MLClientLive*)d)->_port,
                                                    errorMsgL,
                                                    requestInference,
                                                    clearCache,
                                                    ((MLClientLive*)d)->batches,
                                                    b,
                                                    batch_size);

            /* send the request to the server
            ideally we'd skip this after the first batch but it requires a change to the server*/
            comms.sendInferenceRequest(*requestInference);
            MLClientComms::DebugPrint("[LISTENER] listener finished requestInference");

            mlserver::RespondWrapper responseWrapper = ((MLClientLive *) d)->responseWrapper;

            if (comms.readInferenceResponse(responseWrapper)) {
                MLClientComms::DebugPrint("[LISTENER] listener saw readInferenceResponse=1");
                ((MLClientLive *) d)->responseWrapper = responseWrapper;
                ((MLClientLive *) d)->responseStatus = HasReceived;
                ((MLClientLive *) d)->batch = b+1;
                ((MLClientLive *) d)->hashCtr += 1;
                ((MLClientLive *) d)->asapUpdate();
            }
            else{
                MLClientComms::DebugPrint("[LISTENER] listener saw readInferenceResponse=0");
            }
            std::stringstream ss14;
            ss14 << "[LISTENER] batch: " << b << " finished";
            MLClientComms::Print(ss14.str());
        }
        ((MLClientLive *) d)->killthread = true;
    }

    ((MLClientLive *) d)->jobStatus = Completed;
    MLClientComms::Print("[LISTENER] listener exiting");
}

