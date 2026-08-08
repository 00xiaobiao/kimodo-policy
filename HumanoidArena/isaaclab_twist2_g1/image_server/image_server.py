# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
An image server that reads multi-image data from shared memory and publishes it.
Supports ZMQ (default), Redis, DDS, or XRobot remote vision (H264 over TCP).
"""

import base64
import json
import shutil
import socket
import struct
import subprocess
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np
from image_server.shared_memory_utils import MultiImageReader


class _ImagePublisher:
    def publish(self, jpg_bytes: bytes, meta: Dict[str, int]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _ZmqImagePublisher(_ImagePublisher):
    def __init__(self, port: int):
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")

    def publish(self, jpg_bytes: bytes, meta: Dict[str, int]) -> None:
        self.socket.send(jpg_bytes)

    def close(self) -> None:
        self.socket.close()
        self.context.term()


class _RedisImagePublisher(_ImagePublisher):
    def __init__(
            self,
            host: str,
            port: int,
            db: int,
            key_prefix: str,
            channel: str,
    ):
        import redis

        self.client = redis.Redis(host=host, port=port, db=db)
        self.pipeline = self.client.pipeline()
        self.key_jpg = f"{key_prefix}:jpg"
        self.key_meta = f"{key_prefix}:meta"
        self.key_ts = f"{key_prefix}:ts_ms"
        self.channel = channel or ""

    def publish(self, jpg_bytes: bytes, meta: Dict[str, int]) -> None:
        try:
            ts_ms = meta.get("timestamp_ms", int(time.time() * 1000))
            self.pipeline.set(self.key_jpg, jpg_bytes)
            self.pipeline.set(self.key_meta, json.dumps(meta))
            self.pipeline.set(self.key_ts, ts_ms)
            if self.channel:
                self.pipeline.publish(self.channel, str(ts_ms))
            self.pipeline.execute()
        except Exception as exc:
            print(f"[Image Server] Redis publish failed: {exc}")


class _DdsImagePublisher(_ImagePublisher):
    def __init__(self, topic: str):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
            from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
            from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
        except Exception as exc:
            raise RuntimeError(f"DDS imports failed: {exc}") from exc

        try:
            ChannelFactoryInitialize(1)
        except Exception:
            pass

        self.publisher = ChannelPublisher(topic, String_)
        self.publisher.Init()
        self.msg = std_msgs_msg_dds__String_()

    def publish(self, jpg_bytes: bytes, meta: Dict[str, int]) -> None:
        payload = {
            "meta": meta,
            "data": base64.b64encode(jpg_bytes).decode("ascii"),
        }
        self.msg.data = json.dumps(payload)
        self.publisher.Write(self.msg)


class _XRobotImagePublisher(_ImagePublisher):
    def __init__(
            self,
            host: str,
            port: int,
            fps: int,
            bitrate: int,
            target_width: Optional[int],
            target_height: Optional[int],
            ffmpeg_path: Optional[str],
    ):
        self.host = host
        self.port = port
        self.fps = fps
        self.bitrate = bitrate
        self.target_width = target_width
        self.target_height = target_height
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        if not self.ffmpeg_path:
            raise RuntimeError("ffmpeg not found in PATH; set --image_xrobot_ffmpeg")

        self.sock: Optional[socket.socket] = None
        self.proc: Optional[subprocess.Popen] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.running = False
        self.nal_buffer = bytearray()
        self.au_buffer = bytearray()
        self.last_connect_attempt = 0.0
        self.connect_interval = 2.0
        self.current_size = None

        # Performance monitoring
        self.frame_count = 0
        self.encode_times = []
        self.last_perf_report = time.time()

        # Network transmission monitoring
        self.packets_sent = 0
        self.bytes_sent = 0
        self.send_errors = 0
        self.last_network_report = time.time()

    def publish_frame(self, frame) -> None:
        if frame is None:
            return
        frame = self._maybe_resize(frame)
        height, width = frame.shape[:2]
        if not self._ensure_stream(width, height):
            return

        encode_start = time.time()
        try:
            if self.proc and self.proc.stdin:
                self.proc.stdin.write(frame.tobytes())
                self.frame_count += 1

                # Track encoding time
                encode_time = (time.time() - encode_start) * 1000  # ms
                self.encode_times.append(encode_time)
                if len(self.encode_times) > 100:
                    self.encode_times.pop(0)

                # Report performance every 5 seconds
                if time.time() - self.last_perf_report >= 5.0:
                    avg_encode = sum(self.encode_times) / len(self.encode_times)
                    max_encode = max(self.encode_times)
                    print(f"[XRobot] 🎬 Encoding: avg={avg_encode:.2f}ms, max={max_encode:.2f}ms, frames={self.frame_count}")
                    self.last_perf_report = time.time()
        except Exception as exc:
            print(f"[Image Server] XRobot ffmpeg write failed: {exc}")
            self._reset()

    def close(self) -> None:
        self._reset()

    def _ensure_stream(self, width: int, height: int) -> bool:
        if self.current_size != (width, height):
            self._reset()
            self.current_size = (width, height)
        if not self._ensure_socket():
            return False
        if self.proc is None:
            self._start_ffmpeg(width, height)
        return self.proc is not None

    def _ensure_socket(self) -> bool:
        if self.sock and self._socket_ok():
            return True
        now = time.time()
        if now - self.last_connect_attempt < self.connect_interval:
            return False
        self.last_connect_attempt = now
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=2.0)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.sock.settimeout(None)
            print(f"[Image Server] XRobot connected to {self.host}:{self.port}")
            return True
        except Exception as exc:
            print(f"[Image Server] XRobot connect failed: {exc}")
            resolved_ips = []
            try:
                # Resolve DNS / hostname to one or more IPs for better debugging.
                for info in socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM):
                    ip = info[4][0]
                    if ip not in resolved_ips:
                        resolved_ips.append(ip)
            except Exception:
                pass

            ip_hint = f" (resolved: {', '.join(resolved_ips)})" if resolved_ips else ""
            print(
                f"[Image Server] XRobot connect failed to {self.host}:{self.port}{ip_hint}: {exc}"
            )
            self.sock = None
            return False

    def _socket_ok(self) -> bool:
        try:
            return self.sock is not None and self.sock.fileno() >= 0
        except Exception:
            return False

    def _start_ffmpeg(self, width: int, height: int) -> None:
        bitrate = max(100_000, int(self.bitrate))
        fps = max(1, int(self.fps))
        gop = fps
        cmd = [
            self.ffmpeg_path,
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            str(bitrate),
            "-maxrate",
            str(bitrate),
            "-bufsize",
            str(bitrate * 2),
            "-x264-params",
            "repeat-headers=1:aud=1",
            "-f",
            "h264",
            "-",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception as exc:
            print(f"[Image Server] XRobot ffmpeg start failed: {exc}")
            self.proc = None
            return

        self.running = True
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()

    def _read_loop(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        while self.running:
            # Check if proc is still valid before reading
            if not self.proc:
                break
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                # Check if proc is still valid before polling
                if self.proc and self.proc.poll() is not None:
                    break
                time.sleep(0.01)
                continue
            self.nal_buffer.extend(chunk)
            self._drain_nals()

    def _drain_nals(self) -> None:
        while True:
            start = self._find_start_code(self.nal_buffer, 0)
            if start < 0:
                if len(self.nal_buffer) > 4:
                    del self.nal_buffer[:-4]
                return
            next_start = self._find_start_code(self.nal_buffer, start + 3)
            if next_start < 0:
                if start > 0:
                    del self.nal_buffer[:start]
                return
            nal = bytes(self.nal_buffer[start:next_start])
            del self.nal_buffer[:next_start]
            nal_type = self._get_nal_type(nal)
            if nal_type == 9:  # AUD, flush previous access unit
                if self.au_buffer:
                    self._send_packet(bytes(self.au_buffer))
                    self.au_buffer.clear()
                self.au_buffer.extend(nal)
            else:
                self.au_buffer.extend(nal)
                if len(self.au_buffer) > 1_500_000:
                    self._send_packet(bytes(self.au_buffer))
                    self.au_buffer.clear()

    def _find_start_code(self, data: bytearray, start: int) -> int:
        i = start
        end = len(data) - 3
        while i <= end:
            if data[i] == 0 and data[i + 1] == 0:
                if data[i + 2] == 1:
                    return i
                if i + 3 < len(data) and data[i + 2] == 0 and data[i + 3] == 1:
                    return i
            i += 1
        return -1

    def _get_nal_type(self, nal: bytes) -> int:
        if len(nal) < 5:
            return -1
        if nal[0:3] == b"\x00\x00\x01":
            return nal[3] & 0x1F
        if len(nal) >= 6 and nal[0:4] == b"\x00\x00\x00\x01":
            return nal[4] & 0x1F
        return -1

    def _send_packet(self, payload: bytes) -> None:
        if not self._socket_ok():
            return
        try:
            header = struct.pack(">I", len(payload))
            self.sock.sendall(header + payload)

            # Track transmission
            self.packets_sent += 1
            self.bytes_sent += len(header) + len(payload)

            # Report network stats every 3 seconds
            if time.time() - self.last_network_report >= 3.0:
                elapsed = time.time() - self.last_network_report
                mbps = (self.bytes_sent * 8 / 1000000) / elapsed
                avg_packet_size = self.bytes_sent / self.packets_sent if self.packets_sent > 0 else 0
                print(f"[XRobot] 📡 Network: {mbps:.2f} Mbps, packets={self.packets_sent}, avg_size={avg_packet_size/1024:.1f}KB, errors={self.send_errors}")
                self.last_network_report = time.time()
                self.bytes_sent = 0
                self.packets_sent = 0
                self.send_errors = 0
        except Exception as exc:
            self.send_errors += 1
            print(f"[Image Server] XRobot send failed: {exc}")
            self._reset()

    def _maybe_resize(self, frame):
        if self.target_width and self.target_height:
            if (frame.shape[1], frame.shape[0]) != (self.target_width, self.target_height):
                return cv2.resize(frame, (self.target_width, self.target_height))
        return frame

    def _reset(self) -> None:
        self.running = False
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
        if (
                self.reader_thread
                and self.reader_thread.is_alive()
                and self.reader_thread is not threading.current_thread()
        ):
            self.reader_thread.join(timeout=1.0)
        self.reader_thread = None
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None
        self.nal_buffer.clear()
        self.au_buffer.clear()


class ImageServer:
    def __init__(
            self,
            fps=30,
            port=5555,
            Unit_Test=False,
            transport: str = "zmq",
            redis_host: str = "localhost",
            redis_port: int = 6379,
            redis_db: int = 0,
            redis_key_prefix: str = "isaac_image",
            redis_channel: str = "",
            dds_topic: str = "rt/isaac_image",
            xrobot_host: str = "127.0.0.1",
            xrobot_port: int = 12345,
            xrobot_bitrate: int = 4_000_000,
            xrobot_width: Optional[int] = None,
            xrobot_height: Optional[int] = None,
            xrobot_ffmpeg: Optional[str] = None,
            camera_name: str = "head",  # NEW: specify which camera to stream ("head" or "world")
    ):
        """
        Multi-image server - read multi-image data from shared memory and publish it

        Args:
            camera_name: which camera to stream - "head" for front camera, "world" for third-person camera
        """
        print(f"[Image Server] Initializing multi-image server from shared memory (camera: {camera_name})")

        self.fps = fps
        self.port = port
        self.Unit_Test = Unit_Test
        self.transport = transport
        self.camera_name = camera_name  # Store the camera name
        self.running = False
        self.publish_thread = None
        self.frame_count = 0

        # Initialize multi-image shared memory reader
        self.multi_image_reader = MultiImageReader()

        # Set publisher transport
        self.publisher = self._create_publisher(
            transport=transport,
            port=port,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            redis_key_prefix=redis_key_prefix,
            redis_channel=redis_channel,
            dds_topic=dds_topic,
            xrobot_host=xrobot_host,
            xrobot_port=xrobot_port,
            xrobot_bitrate=xrobot_bitrate,
            xrobot_width=xrobot_width,
            xrobot_height=xrobot_height,
            xrobot_ffmpeg=xrobot_ffmpeg,
        )

        if self.Unit_Test:
            self._init_performance_metrics()

        print(f"[Image Server] Multi-image server initialized ({self.transport})")

        # start the publishing thread
        self.start_publishing()

    def _create_publisher(
            self,
            transport: str,
            port: int,
            redis_host: str,
            redis_port: int,
            redis_db: int,
            redis_key_prefix: str,
            redis_channel: str,
            dds_topic: str,
            xrobot_host: str,
            xrobot_port: int,
            xrobot_bitrate: int,
            xrobot_width: Optional[int],
            xrobot_height: Optional[int],
            xrobot_ffmpeg: Optional[str],
    ) -> _ImagePublisher:
        if transport == "zmq":
            return _ZmqImagePublisher(port)
        if transport == "redis":
            return _RedisImagePublisher(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                key_prefix=redis_key_prefix,
                channel=redis_channel,
            )
        if transport == "dds":
            return _DdsImagePublisher(dds_topic)
        if transport == "xrobot":
            return _XRobotImagePublisher(
                host=xrobot_host,
                port=xrobot_port,
                fps=self.fps,
                bitrate=xrobot_bitrate,
                target_width=xrobot_width,
                target_height=xrobot_height,
                ffmpeg_path=xrobot_ffmpeg,
            )
        raise ValueError(f"Unsupported image transport: {transport}")

    def _init_performance_metrics(self):
        self.frame_count = 0
        self.time_window = 1.0
        self.frame_times = []
        self.start_time = time.time()

    def _update_performance_metrics(self, current_time):
        self.frame_times.append(current_time)
        # Remove timestamps outside the time window
        self.frame_times = [t for t in self.frame_times if t >= current_time - self.time_window]
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window if self.frame_times else 0
            print(
                f"[Image Server] Real-time FPS: {real_time_fps:.2f}, Total frames sent: {self.frame_count}, Elapsed time: {elapsed_time:.2f} sec")

    def send_process(self):
        """Read the concatenated images from shared memory and send them"""
        print("[Image Server] Starting send_process from shared memory...")
        print(f"[Image Server] Streaming {self.camera_name} camera view")

        # Map camera_name to the key used in read_images()
        camera_key_map = {
            "head": "head",
            "front_camera": "head",
            "world": "world",
            "world_camera": "world",
            "left": "left",
            "left_wrist_camera": "left",
            "right": "right",
            "right_wrist_camera": "right",
        }

        camera_key = camera_key_map.get(self.camera_name, "head")
        print(f"[Image Server] Using camera key: {camera_key}")
        print(f"[Image Server] ⚙️  FPS Configuration: target={self.fps} fps, transport={self.transport}")

        # FPS monitoring variables
        fps_start_time = time.time()
        fps_frame_count = 0
        fps_report_interval = 2.0  # Report FPS every 2 seconds (changed from 5.0 for faster feedback)

        # Frame change detection
        last_frame_hash = None
        duplicate_count = 0
        unique_count = 0

        try:
            if not self.running:
                self.running = True
            while self.running:
                # read all camera images from shared memory
                images = self.multi_image_reader.read_images()

                if images is None or len(images) == 0:
                    # if there is no image data, wait a moment and try again
                    time.sleep(0.01)
                    continue

                # extract the specific camera
                if camera_key not in images:
                    # if the requested camera is not available, wait and try again
                    time.sleep(0.01)
                    continue

                camera_image = images[camera_key]

                # Detect frame changes using hash
                import hashlib
                frame_hash = hashlib.md5(camera_image.tobytes()).hexdigest()
                if frame_hash == last_frame_hash:
                    duplicate_count += 1
                else:
                    unique_count += 1
                    last_frame_hash = frame_hash

                # Extract depth data if available
                depth_key = f"{camera_key}_depth"
                camera_depth = images.get(depth_key, None)

                # show the concatenated images
                # cv2.imshow(f'{self.camera_name} Camera View', camera_image)
                # key = cv2.waitKey(1) & 0xFF
                # if key == ord('q') or key == 27:  # 'q' 或 ESC 键退出
                #     print("[Image Server] User pressed quit key")
                #     break

                if self.transport == "xrobot":
                    self.publisher.publish_frame(camera_image)
                else:
                    # encode the images
                    ret, buffer = cv2.imencode('.jpg', camera_image)
                    if not ret:
                        print("[Image Server] Frame imencode is failed.")
                        continue

                    jpg_bytes = buffer.tobytes()

                    # Prepare depth data if available
                    depth_bytes = b''
                    if camera_depth is not None:
                        # Ensure depth is float32 and contiguous
                        if camera_depth.dtype != np.float32:
                            camera_depth = camera_depth.astype(np.float32)
                        if not camera_depth.flags['C_CONTIGUOUS']:
                            camera_depth = np.ascontiguousarray(camera_depth)
                        depth_bytes = camera_depth.tobytes()

                    # Pack message: [width][height][jpeg_len][jpeg_data][depth_len][depth_data]
                    height, width, channels = camera_image.shape
                    import struct
                    header = struct.pack('iii', width, height, len(jpg_bytes))
                    depth_header = struct.pack('i', len(depth_bytes))

                    # Combine all parts
                    message = header + jpg_bytes + depth_header + depth_bytes

                    # Send the combined message
                    meta = {
                        "timestamp_ms": int(time.time() * 1000),
                        "height": int(height),
                        "width": int(width),
                        "channels": int(channels),
                        "format": "jpg",
                        "has_depth": len(depth_bytes) > 0,
                    }
                    self.publisher.publish(message, meta)
                self.frame_count += 1
                fps_frame_count += 1

                # Print progress every 50 frames
                if self.frame_count % 50 == 0:
                    print(f"[Image Server] 📊 Sent {self.frame_count} frames total")

                # Report FPS periodically
                current_time = time.time()
                elapsed = current_time - fps_start_time
                if elapsed >= fps_report_interval:
                    actual_fps = fps_frame_count / elapsed
                    unique_fps = unique_count / elapsed
                    duplicate_ratio = duplicate_count / (duplicate_count + unique_count) * 100 if (duplicate_count + unique_count) > 0 else 0
                    print(f"[Image Server] 🎯 Actual FPS: {actual_fps:.2f} (target: {self.fps}, frames sent: {fps_frame_count})")
                    print(f"[Image Server] 🔄 Frame changes: unique={unique_count} ({unique_fps:.2f} FPS), duplicates={duplicate_count} ({duplicate_ratio:.1f}%)")
                    fps_start_time = current_time
                    fps_frame_count = 0
                    duplicate_count = 0
                    unique_count = 0

                if self.fps and self.fps > 0:
                    time.sleep(1.0 / self.fps)

        except KeyboardInterrupt:
            print("[Image Server] Interrupted by user.")
        finally:
            cv2.destroyAllWindows()  # close the display window
            self._close()

    def start_publishing(self):
        """Start the publishing thread"""
        if not self.running:
            self.running = True
            self.publish_thread = threading.Thread(target=self.send_process)
            self.publish_thread.daemon = True
            self.publish_thread.start()
            print("[Image Server] Multi-image publishing thread started")

    def stop_publishing(self):
        """Stop the publishing thread"""
        if hasattr(self, 'running') and self.running:
            self.running = False
            # Only join if we're not in the publish thread itself
            if self.publish_thread and self.publish_thread is not threading.current_thread():
                self.publish_thread.join(timeout=1.0)
            print("[Image Server] Publishing thread stopped")

    def _close(self):
        """Close the server"""
        self.stop_publishing()
        cv2.destroyAllWindows()

        # close the shared memory reader
        if hasattr(self, 'multi_image_reader'):
            self.multi_image_reader.close()

        # close the network connection
        if hasattr(self, "publisher") and self.publisher:
            self.publisher.close()
        print("[Image Server] Multi-image server closed")

    def __del__(self):
        """Destructor"""
        self._close()


if __name__ == "__main__":
    # use the send_process mode example
    server = ImageServer(fps=30, Unit_Test=False)

    # use the send_process method (blocking)
    server.send_process()

    # or use the thread mode
    # try:
    #     server.start_publishing()
    #     print("[Image Server] Server running... Press Ctrl+C to stop")
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("\n[Image Server] Interrupted by user")
    # finally:
    #     server._close()

