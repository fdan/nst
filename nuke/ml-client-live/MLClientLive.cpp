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
    Status status = ReadyToSend;
    batch = 0;
}

MLClientLive::~MLClientLive() {
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


void MLClientLive::renderStripe(ImagePlane& imagePlane)
{
  std::thread::id this_id = std::this_thread::get_id();

//    std::stringstream  ss111;
//    ss111 << "renderStripe saw status " << status << std::endl;
//    MLClientComms::Vprint(ss111.str());

  // get the knob values.  floats?
  DD::Image::Knob* itKnob = knob("iterations");
  DD::Image::Knob* bsKnob = knob("batch_size");
  iterations = itKnob->get_value();

  // need to + 1?
  batch_size = bsKnob->get_value();
  batches = iterations / batch_size;

  if (aborted() || cancelled()) {
    std::stringstream  ss112;
    ss112 << "renderStripe aborted or cancelled";
    MLClientComms::Vprint(ss112.str());
    return;
  }

  if (haveValidModelInfo() && _modelSelected) {

//      std::stringstream  ss113;
//      ss113 << "renderStripe go";
//      MLClientComms::Vprint(ss113.str());

    std::string errorMsg;

      if (status==ReadyToSend){
          std::stringstream  ss1;
          ss1 << "renderStripe saw status ReadyToSend, sending " << this_id;
          MLClientComms::Vprint(ss1.str());
//          status = Sending;
//          status = HasReceived;
          Thread::spawn(::listener, 1, this);
          std::stringstream  ss2;
          ss2 << "renderStripe spawned listener, returning " << this_id;
          MLClientComms::Vprint(ss2.str());

          // simple pass through to avoid erroring out
          input0().fetchPlane(imagePlane);
          return;
      }

    if (status==Sending){
        std::stringstream ss3;
        ss3 << "renderStripe saw status: Sending, returning " << this_id;
        MLClientComms::Vprint(ss3.str());
        return;
    }

    if (status==Waiting){
        std::stringstream ss4;
        ss4 << "renderStripe saw status: Waiting, returning " << this_id;
        MLClientComms::Vprint(ss4.str());
        return;
    }

    if (status==HasReceived) {
        std::stringstream ss5;
        ss5 << "renderStripe saw status: HasReceived, processing result " << this_id;
        MLClientComms::Vprint(ss5.str());

        if (!fillImagePlane(imagePlane, errorMsg)) {
            MLClientComms::Vprint(errorMsg);
            error(errorMsg.c_str());
        }
    }

    status = ReadyToSend;
    killthread = false;
    batch = 0;

    return;
  }

  // Check again if we hit abort during processing
  if (aborted() || cancelled()) {
    MLClientComms::Vprint("Aborted without processing image.");
    return;
  }

  // If we reached here by default let's pull an image from input0() so
  // that it's at least passing something through.
  input0().fetchPlane(imagePlane);
}

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
  ss << "Server can serve " << std::to_string(numModels) << " models" << std::endl;
  ss << "-----------------------------------------------";
  MLClientComms::Vprint(ss.str());
  for (int i = 0; i < numModels; i++) {
    mlserver::Model m;
    m = responseWrapper.r1().models(i);
    modelNames.push_back(m.label());
    _serverModels.push_back(m);
    _numInputs.push_back(m.inputs_size());
    std::vector<std::string> names;
    for (int j = 0; j < m.inputs_size(); j++) {
      mlserver::ImagePrototype p;
      p = m.inputs(j);
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

// todo: add clearCache to request and method signature
bool MLClientLive::getInferenceRequest(const std::string& hostStr,
                                       int port,
                                       std::string& errorMsg,
                                       mlserver::RequestInference* requestInference,
                                       bool clearCache,
                                       int batch_total,
                                       int batch_current){

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

    // todo: the opt image is the first input.  get from cached value
    for (int i = 0; i < node_inputs(); i++) {

        const ChannelSet readChannels = input(i)->info().channels();

        // Create an ImagePlane, and read each input into it.
        // Get our input & sanity check
        DD::Image::Iop *inputIop = dynamic_cast<DD::Image::Iop *>(input(i));
        if (inputIop == NULL) {
            errorMsg = "Input is empty or not connected.";
            return false;
        }

        // Checking before validating inputs
        if (aborted()) {
            errorMsg = "Process aborted before validating inputs.";
            return false;
        }

        // Try validate & request the input, this should be quick if the data
        // has already been requested.
        if (!inputIop->tryValidate(/*force*/true)) {
            errorMsg = "Unable to validate input.";
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
            errorMsg = "No image data fetched from input.";
            return false;
        }

        // Checking after fetching inputs
        if (aborted()) {
            errorMsg = "Process aborted after fetching inputs.";
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
            errorMsg = "Image size is zero.";
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
  static const char* static_choices[] = {
      0};
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
}

int MLClientLive::knob_changed(Knob* knobChanged)
{
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

//  // Check if dynamic button is pressed
//  for (int i = 0; i < getModelManager().getNumOfButtons(); i++) {
//    if (knobChanged->is(getModelManager().getDynamicButtonName(i).c_str())) {
//      // Set current button to true (pressed) for model inference
//      getModelManager().setDynamicButtonValue(i, true);
//      // Set up our error string
//      std::string errorMsg;
//      // Set up our incoming response message structure.
//      mlserver::RespondWrapper responseWrapper;
//      // Wrap up our image data to be sent, send it, and
//      // retrieve the response.
//      if (!processImage(_host, _port, responseWrapper, errorMsg)) {
//        error(errorMsg.c_str());
//      }
//
//      // Get the resulting general data
//      if (responseWrapper.has_r2() && responseWrapper.r2().num_objects() > 0) {
//        const mlserver::FieldValuePairAttrib object = responseWrapper.r2().objects(0);
//        // Run script in Nuke if object called PythonScript is created
//        if (object.name() == "PythonScript") {
//          // Check object has string_attributes
//          if (object.values_size() != 0
//            && object.values(0).string_attributes_size() != 0) {
//            mlserver::StringAttrib pythonScript = object.values(0).string_attributes(0);
//            // Run Python Script in Nuke
//            if (pythonScript.values_size() != 0) {
//              script_command(pythonScript.values(0).c_str(), true, false);
//              script_unlock();
//            }
//          }
//        }
//      }
//      // Set current button to false (unpressed)
//      getModelManager().setDynamicButtonValue(i, false);
//      return 1;
//    }
//  }
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

    MLClientComms::Vprint("\nlistener doing stuff...");

    while (!((MLClientLive *) d)->killthread) {

        bool clearCache = true;

        for (int b = 0; b < ((MLClientLive*)d)->batches; b++) {
            std::stringstream ss;
            ss << "batch: " << b;
            MLClientComms::Vprint(ss.str());

            MLClientComms comms(((MLClientLive *) d)->_host, ((MLClientLive *) d)->_port);

            std::string errorMsg;

            mlserver::RequestInference* requestInference = new mlserver::RequestInference;

            if (!b==0){
                clearCache = false;
            }

            ((MLClientLive*)d)->getInferenceRequest(((MLClientLive*)d)->_host,
                                                    ((MLClientLive*)d)->_port,
                                                    errorMsg,
                                                    requestInference,
                                                    clearCache,
                                                    ((MLClientLive*)d)->batches,
                                                    b);

            /* send the request to the server
            ideally we'd skip this after the first batch but it requires a change to the server*/
            comms.sendInferenceRequest(*requestInference);

            mlserver::RespondWrapper responseWrapper = ((MLClientLive *) d)->responseWrapper;
            if (comms.readInferenceResponse(responseWrapper)) {
                                ((MLClientLive *) d)->responseWrapper = responseWrapper;
                ((MLClientLive *) d)->status = HasReceived;
                ((MLClientLive *) d)->batch = b+1;
                ((MLClientLive *) d)->hashCtr += 1;
                ((MLClientLive *) d)->asapUpdate();
            }
        }
        ((MLClientLive *) d)->killthread = true;
    }
}

