#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import zmq

from .xlerobot import XLerobot
from .config_xlerobot import XLerobotConfig, XLerobotHostConfig


class ObservationStreamer:
    def __init__(
        self,
        zmq_socket: zmq.Socket,
        camera_keys: tuple[str, ...],
        jpeg_quality: int = 90,
    ) -> None:
        self._socket = zmq_socket
        self._camera_keys = camera_keys
        self._jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[dict[str, object] | None]" = queue.Queue(maxsize=2)
        self._encoder_pool = (
            ThreadPoolExecutor(max_workers=max(1, len(self._camera_keys)))
            if self._camera_keys
            else None
        )
        self._thread = (
            threading.Thread(target=self._run, name="observation-streamer", daemon=True)
            if self._camera_keys
            else None
        )

    def start(self) -> None:
        if self._thread is not None:
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            while True:
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._thread.join()
        if self._encoder_pool is not None:
            self._encoder_pool.shutdown(wait=True)

    def publish(self, observation: dict[str, object]) -> None:
        if not self._camera_keys:
            self._send(observation)
            return
        try:
            self._queue.put_nowait(observation)
        except queue.Full:
            logging.debug("Dropping observation due to encoder backlog")

    def _encode_frame(self, frame) -> str:
        ret, buffer = cv2.imencode(".jpg", frame, self._jpeg_params)
        if not ret:
            return ""
        return base64.b64encode(buffer).decode("utf-8")

    def _encode_observation(self, observation: dict[str, object]) -> dict[str, object]:
        encoded_observation = dict(observation)
        if not self._encoder_pool:
            return encoded_observation
        futures = {}
        for key in self._camera_keys:
            if key not in observation:
                continue
            frame = observation.get(key)
            if frame is None:
                encoded_observation[key] = ""
                continue
            futures[self._encoder_pool.submit(self._encode_frame, frame)] = key
        for future, key in futures.items():
            try:
                encoded_observation[key] = future.result()
            except Exception:
                logging.exception("Failed to encode frame for camera '%s'", key)
                encoded_observation[key] = ""
        return encoded_observation

    def _send(self, observation: dict[str, object]) -> None:
        try:
            self._socket.send_string(json.dumps(observation), flags=zmq.NOBLOCK)
        except zmq.Again:
            logging.info("Dropping observation, no client connected")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                encoded = self._encode_observation(item)
                self._send(encoded)
            finally:
                self._queue.task_done()


class XLerobotHost:
    def __init__(self, config: XLerobotHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def main():
    logging.info("Configuring Xlerobot")
    robot_config = XLerobotConfig(id="my_xlerobot_pc")
    robot = XLerobot(robot_config)

    logging.info("Connecting Xlerobot")
    robot.connect()

    logging.info("Starting HostAgent")
    host_config = XLerobotHostConfig()
    host = XLerobotHost(host_config)

    streamer = ObservationStreamer(
        host.zmq_observation_socket,
        tuple(robot.cameras.keys()),
        jpeg_quality=90,
    )
    streamer.start()

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")
    try:
        # Business logic
        start = time.perf_counter()
        duration = 0
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                _action_sent = robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                if not watchdog_active:
                    logging.warning("No command available")
            except Exception as e:
                logging.error("Message fetching failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                watchdog_active = True
                robot.stop_base()

            last_observation = robot.get_observation()

            streamer.publish(last_observation)

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Lekiwi Host.")
        robot.disconnect()
        streamer.stop()
        host.disconnect()

    logging.info("Finished LeKiwi cleanly")


if __name__ == "__main__":
    main()
