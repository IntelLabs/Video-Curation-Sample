/**
 * @file   DynamicMetadataHandler.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2023 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#include <chrono>
#include "DynamicMetadataHandler.h"
#include "vcl/Exception.h"
#include "../utils/include/kubernetes/KubeHelper.h"
#include "VideoLoop.h"

DynamicMetadataHandler::DynamicMetadataHandler(VCL::Video video, Json::Value props, std::vector<Json::Value> metadata){
    _video = video;
    _metadata = metadata;
    _props = props;
}

void DynamicMetadataHandler::initiate(){
    add_metadata_bg_vid();
}

void DynamicMetadataHandler::add_metadata_bg_vid(){
    // std::string url = "http://udf-service:5011/video";
    std::string url = "http://udf-bkgd-service:5012/video";
    Json::Value options;
    // options["port"] = VDMSConfig::instance()->get_int_value("port", DEFAULT_PORT);
    options["port"] = 55555;
    options["id"] = "add_metadata";
    options["uid"] = _props[VDMS_BG_UNIQUE_VID_ID];

    int frame_count = _video.get_frame_count();
    int num_bbs = _metadata[0].size();
    int curr_frame = 0;
    int bb_idx = 0;
    int counter = 0;
    int bb_counter = 0;
    int desired_chunk_size = 50;
    int chunk_count = static_cast<int>(std::ceil(static_cast<double>(num_bbs) / desired_chunk_size));  // std::ceil(num_bbs / desired_chunk_size);
    VideoLoop videoLoop;
    videoLoop.set_nrof_entities(chunk_count);

    std::cout << "[DEBUG] _metadata[0] size: " << num_bbs << std::endl;

    // TODO: Frames are out of order here
    // std::cout << "[DEBUG] add_metadata_bg_vid _metadata: " << _metadata[0] << std::endl;

    Json::Value metadata_chunk;
    for (Json::Value vframe : _metadata[0]) {
        // curr_frame++;
        Json::Value curr_frame_metadata;
        Json::Value frame_props;
        frame_props[VDMS_DM_VID_IDX_PROP] = vframe["frameId"].asInt();
        frame_props[VDMS_DM_VID_NAME_PROP] = _props[VDMS_VID_PATH_PROP];
        frame_props["server_filepath"] = _props["Name"].asString();
        // frame_props["fps"] = _props["fps"].asFloat();
        // frame_props["duration"] = _props["duration"].asFloat();
        // frame_props["width"] = _props["width"].asFloat();
        // frame_props["height"] = _props["height"].asFloat();
        // frame_props["frame_count"] = _props["frame_count"].asInt();

        curr_frame_metadata["frame_props"] = frame_props;

        Json::Value edge_props;
        edge_props[VDMS_DM_VID_IDX_PROP] = vframe["frameId"].asInt();
        edge_props[VDMS_DM_VID_NAME_PROP] = _props[VDMS_VID_PATH_PROP];
        edge_props["server_filepath"] = _props["Name"].asString();
        // edge_props["fps"] = _props["fps"].asFloat();
        // edge_props["duration"] = _props["duration"].asFloat();
        // edge_props["width"] = _props["width"].asFloat();
        // edge_props["height"] = _props["height"].asFloat();
        // edge_props["frame_count"] = _props["frame_count"].asInt();

        curr_frame_metadata["edge_props"] = edge_props;

        if (vframe.isMember("bbox")) {
            Json::Value bbox_props;
            bbox_props[VDMS_DM_VID_IDX_PROP] = vframe["frameId"].asInt();
            bbox_props["server_filepath"] = _props["Name"].asString();
            // bbox_props["fps"] = _props["fps"].asFloat();
            // bbox_props["duration"] = _props["duration"].asFloat();
            // bbox_props["width"] = _props["width"].asFloat();
            // bbox_props["height"] = _props["height"].asFloat();
            // bbox_props["frame_count"] = _props["frame_count"].asInt();
            bbox_props[VDMS_DM_VID_NAME_PROP] = _props[VDMS_VID_PATH_PROP];
            bbox_props[VDMS_DM_VID_OBJECT_PROP] =
                vframe["bbox"]["object"].asString();
            bbox_props[VDMS_ROI_COORD_X_PROP] = vframe["bbox"]["x"].asFloat();
            bbox_props[VDMS_ROI_COORD_Y_PROP] = vframe["bbox"]["y"].asFloat();
            bbox_props[VDMS_ROI_WIDTH_PROP] = vframe["bbox"]["width"].asFloat();
            bbox_props[VDMS_ROI_HEIGHT_PROP] = vframe["bbox"]["height"].asFloat();

            for (auto member : vframe["bbox"]["object_det"].getMemberNames()) {
                if (member == "age")
                bbox_props[member] = vframe["bbox"]["object_det"][member].asInt();
                if (member == "confidence")
                bbox_props[member] = vframe["bbox"]["object_det"][member].asFloat();
                if (member == "gender" || member == "emotion")
                bbox_props[member] = vframe["bbox"]["object_det"][member].asString();
                if (member == "frameW" || member == "frameH")
                bbox_props[member] = vframe["bbox"]["object_det"][member].asInt();
            }

            curr_frame_metadata["bbox_props"] = bbox_props;

            Json::Value bb_edge_props;
            bb_edge_props[VDMS_DM_VID_IDX_PROP] = vframe["frameId"].asInt();
            bb_edge_props[VDMS_DM_VID_NAME_PROP] = _props[VDMS_VID_PATH_PROP];
            bb_edge_props["server_filepath"] = _props["Name"].asString();
            // bb_edge_props["fps"] = _props["fps"].asFloat();
            // bb_edge_props["duration"] = _props["duration"].asFloat();
            // bb_edge_props["width"] = _props["width"].asFloat();
            // bb_edge_props["height"] = _props["height"].asFloat();
            // bb_edge_props["frame_count"] = _props["frame_count"].asInt();

            curr_frame_metadata["bb_edge_props"] = bb_edge_props;

            metadata_chunk[vframe["bbId"].asString()] = curr_frame_metadata;
            // metadata_chunk[vframe["frameId"].asString()] = curr_frame_metadata;
            bb_counter++;
            bb_idx++;

        }
        counter++;
        // std::cout << "[DEBUG Dynamic] frame_count: " << std::to_string(frame_count) << std::endl;
        // std::cout << "[DEBUG Dynamic] curr_frame: " << std::to_string(curr_frame) << std::endl;
        // std::cout << "[DEBUG Dynamic] bb_idx: " << std::to_string(bb_idx) << std::endl;
        // std::cout << "[DEBUG Dynamic] int(frame_count/chunk_count): " << std::to_string(int(frame_count/chunk_count)) << std::endl;
        // std::cout << "[DEBUG Dynamic] int(num_bbs/chunk_count): " << std::to_string(int(num_bbs/chunk_count)) << std::endl;
        // std::cout << "[DEBUG] add_metadata_bg_vid counter: " << std::to_string(counter) << std::endl;
        // if (counter == int(frame_count/chunk_count) || curr_frame == frame_count){
        if (bb_counter == int(num_bbs/chunk_count) || bb_idx == num_bbs){
            options["metadata"] = metadata_chunk;
            options["Name"] = _props["Name"].asString();
            VCL::Video video(_video);
            // std::cout << "[DEBUG] add_metadata_bg_vid counter (" << std::to_string(counter) << ") for Video id: " << video.get_video_id() << std::endl;
            // std::cout << "[DEBUG] add_metadata_bg_vid bb_counter (" << std::to_string(bb_counter) << ") for Video id: " << video.get_video_id() << std::endl;
            video.remoteOperation(url, options);
            videoLoop.enqueue(video);
            metadata_chunk.clear();
            counter = 0;
            bb_counter = 0;
        }
    }

    while (videoLoop.is_loop_running()) {
      continue;
    }
}