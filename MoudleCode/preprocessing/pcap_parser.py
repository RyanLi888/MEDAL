"""
PCAP Parser Module
Extracts 4-dimensional features from PCAP files
"""
import numpy as np
import os
import glob
import re
from scapy.all import rdpcap, IP, TCP
from scapy.error import Scapy_Exception
import logging

try:
    from MoudleCode.utils.config import config
except Exception:
    config = None

logger = logging.getLogger(__name__)


def extract_flow_count(filename):
    """
    Extract flow count from filename
    
    Filename patterns:
    - flow_250.pcap -> 250
    - benign_100.pcap -> 100
    - malware_50.pcapng -> 50
    - Any number in filename
    
    Args:
        filename: PCAP filename
        
    Returns:
        flow_count: int, or None if not found
    """
    # Try to find numbers in filename
    numbers = re.findall(r'\d+', filename)
    
    if numbers:
        # Use the largest number found (likely the flow count)
        return int(max(numbers, key=lambda x: int(x)))
    
    return None


class PCAPParser:
    """Parser for extracting features from PCAP files"""
    
    def __init__(self, sequence_length=1024, max_packets_to_extract=5000):
        """
        Args:
            sequence_length: Maximum number of packets to extract (L=1024)
            max_packets_to_extract: Maximum packets to process per flow before feature extraction (default: 5000)
        """
        self.sequence_length = sequence_length
        self.max_packets_to_extract = max_packets_to_extract
        
    def _extract_flow_key(self, pkt):
        """
        Extract flow key (5-tuple) from packet
        
        Args:
            pkt: Scapy packet
            
        Returns:
            flow_key: tuple (src_ip, dst_ip, src_port, dst_port, protocol) or None
        """
        if IP not in pkt:
            return None
        
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        protocol = pkt[IP].proto
        
        src_port = 0
        dst_port = 0
        
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif hasattr(pkt[IP], 'payload') and hasattr(pkt[IP].payload, 'sport'):
            # UDP or other protocols
            try:
                src_port = pkt[IP].payload.sport
                dst_port = pkt[IP].payload.dport
            except:
                pass
        
        # Normalize flow key (smaller IP/port first for bidirectional flows)
        if src_ip < dst_ip or (src_ip == dst_ip and src_port < dst_port):
            return (src_ip, dst_ip, src_port, dst_port, protocol)
        else:
            return (dst_ip, src_ip, dst_port, src_port, protocol)
    
    def parse_pcap_file(self, pcap_path, extract_flows=True, flow_timeout=120.0):
        """
        Parse a single PCAP file and extract 4D features
        
        If extract_flows=True, extracts multiple flows from the PCAP file.
        Each flow is identified by 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol).
        Packets with the same 5-tuple are grouped into the same flow.
        If packet interval exceeds flow_timeout seconds, a new flow is started.
        
        Args:
            pcap_path: Path to PCAP file
            extract_flows: If True, extract multiple flows; if False, treat entire file as one flow
            flow_timeout: Timeout in seconds for flow splitting (default: 120s)
            
        Returns:
            If extract_flows=True: list of feature arrays (one per flow)
            If extract_flows=False: single feature array of shape (L, 4)
        """
        try:
            packets = rdpcap(pcap_path)
        except (Scapy_Exception, FileNotFoundError) as e:
            logger.error(f"Error reading {pcap_path}: {e}")
            return None
        
        if len(packets) == 0:
            logger.warning(f"No packets in {pcap_path}")
            return None
        
        if not extract_flows:
            # Original behavior: treat entire file as one flow
            return self._parse_single_flow(packets)
        
        # Group packets by flow (5-tuple) with timeout-based splitting
        # flows: dict mapping (flow_key, flow_id) -> list of packets
        # flow_state: dict mapping flow_key -> (last_time, current_flow_id)
        flows = {}
        flow_state = {}
        
        for pkt in packets:
            flow_key = self._extract_flow_key(pkt)
            if flow_key is None:
                continue
            
            pkt_time = float(pkt.time)
            
            # Check if this flow_key has been seen before
            if flow_key in flow_state:
                last_time, current_flow_id = flow_state[flow_key]
                
                # Check if packet interval exceeds timeout
                if pkt_time - last_time > flow_timeout:
                    # Start a new flow with incremented flow_id
                    current_flow_id += 1
                    logger.debug(f"Flow timeout detected for {flow_key}: {pkt_time - last_time:.2f}s > {flow_timeout}s, starting new flow")
                
                # Update flow state
                flow_state[flow_key] = (pkt_time, current_flow_id)
            else:
                # First packet of this flow_key
                current_flow_id = 0
                flow_state[flow_key] = (pkt_time, current_flow_id)
            
            # Add packet to the appropriate flow
            flow_identifier = (flow_key, current_flow_id)
            if flow_identifier not in flows:
                flows[flow_identifier] = []
            flows[flow_identifier].append(pkt)
        
        if len(flows) == 0:
            logger.warning(f"No valid flows found in {pcap_path}")
            return []
        
        logger.info(f"在 {os.path.basename(pcap_path)} 中发现 {len(flows)} 条流（基于5元组+{flow_timeout}秒超时），开始提取4维特征...")
        logger.info(f"  每个数据包提取: [Length, Direction, BurstSize, ValidMask]")
        logger.info(f"  每个流转换为: 1024×4 的特征序列")
        logger.info("")
        
        # Extract features for each flow
        all_flow_features = []
        flow_count = 0
        
        for flow_identifier, flow_packets in flows.items():
            flow_count += 1
            flow_features = self._parse_single_flow(flow_packets)
            
            if flow_features is not None:
                all_flow_features.append(flow_features)
                # 每10条流或最后一条流输出详细日志
                if flow_count % 10 == 0 or flow_count == len(flows):
                    logger.info(f"  ✓ 完成 {flow_count}/{len(flows)} 条流 (当前流包含 {len(flow_packets)} 个数据包，已提取 {flow_features.shape[0]}×4 特征)")
            else:
                logger.warning(f"  ✗ 流 {flow_count}/{len(flows)} 解析失败")
        
        logger.info("")
        logger.info(f"✓ 成功提取 {len(all_flow_features)}/{len(flows)} 条流的4维特征序列")
        
        return all_flow_features if len(all_flow_features) > 0 else []
    
    def _parse_single_flow(self, packets):
        """
        Parse a single flow (list of packets) and extract 4D features
        
        Args:
            packets: List of Scapy packets belonging to one flow
            
        Returns:
            features: numpy array of shape (L, 4)
                     [Length, Direction, BurstSize, ValidMask]
        """
        if len(packets) == 0:
            return None
        
        # Limit packets to max_packets_to_extract for performance
        if len(packets) > self.max_packets_to_extract:
            logger.debug(f"Flow has {len(packets)} packets, limiting to first {self.max_packets_to_extract} for feature extraction")
            packets = packets[:self.max_packets_to_extract]
        
        # Sort packets by time
        packets = sorted(packets, key=lambda p: float(p.time))
        
        # Extract features
        length_norm_list = []
        direction_list = []
        raw_length_list = []
        raw_iat_list = []
        prev_time = None
        
        # Determine flow direction (use first packet with IP layer)
        client_ip = None
        server_ip = None
        for pkt in packets:
            if IP in pkt:
                client_ip = pkt[IP].src
                server_ip = pkt[IP].dst
                break
        
        if client_ip is None:
            return None
        
        for pkt in packets:
            if IP not in pkt:
                continue
            
            # Feature 0: Length (normalized by 1500)
            length = len(pkt)
            raw_length_list.append(float(length))
            length_norm = min(length / 1500.0, 1.0)
            
            # Use IAT only for burst boundary detection (not an output feature)
            if prev_time is None:
                raw_iat = 0.0
                prev_time = float(pkt.time)
            else:
                raw_iat = float(pkt.time) - prev_time
                prev_time = float(pkt.time)
            
            # Feature 2: Direction (+1: client->server, -1: server->client)
            if pkt[IP].src == client_ip:
                direction = 1.0
            else:
                direction = -1.0
            
            length_norm_list.append(float(length_norm))
            direction_list.append(float(direction))
            raw_iat_list.append(float(raw_iat))
        
        if len(length_norm_list) == 0:
            return None

        raw_lengths = np.asarray(raw_length_list, dtype=np.float32)
        directions = np.asarray(direction_list, dtype=np.float32)
        raw_iats = np.asarray(raw_iat_list, dtype=np.float32)

        burst_iat_th = 0.1
        try:
            if config is not None and hasattr(config, 'BURST_IAT_THRESHOLD'):
                burst_iat_th = float(getattr(config, 'BURST_IAT_THRESHOLD'))
        except Exception:
            burst_iat_th = 0.1

        burst_size = np.zeros((raw_lengths.shape[0],), dtype=np.float32)
        if raw_lengths.shape[0] > 0:
            start_idx = 0
            current_dir = directions[0]
            current_sum = raw_lengths[0]
            for i in range(1, raw_lengths.shape[0]):
                new_burst = (directions[i] != current_dir) or (raw_iats[i] > burst_iat_th)
                if new_burst:
                    burst_size[start_idx:i] = current_sum
                    start_idx = i
                    current_dir = directions[i]
                    current_sum = raw_lengths[i]
                else:
                    current_sum += raw_lengths[i]
            burst_size[start_idx:] = current_sum
        burst_size = np.log1p(burst_size).astype(np.float32)

        valid_mask = np.ones((raw_lengths.shape[0],), dtype=np.float32)

        # Convert to numpy array
        features = np.column_stack([
            np.asarray(length_norm_list, dtype=np.float32),
            np.asarray(direction_list, dtype=np.float32),
            burst_size,
            valid_mask,
        ]).astype(np.float32)
        
        # Sequence alignment (truncate or pad to L=1024)
        features = self._align_sequence(features)
        
        return features
    
    def _align_sequence(self, features):
        """
        Align sequence to fixed length L
        
        Args:
            features: numpy array of shape (N, 4)
            
        Returns:
            aligned_features: numpy array of shape (L, 4)
        """
        n_packets = features.shape[0]
        
        if n_packets >= self.sequence_length:
            # Truncate: keep first L packets
            return features[:self.sequence_length]
        else:
            # Pad: append zeros
            padding = np.zeros((self.sequence_length - n_packets, 4), dtype=np.float32)
            return np.vstack([features, padding])
    
    def parse_directory(self, directory, label, max_flows=None):
        """
        Parse PCAP files from a directory based on flow count in filename
        
        Args:
            directory: Path to directory containing PCAP files
            label: Label for this directory (0 for benign, 1 for malicious)
            max_flows: Number of flows needed (selects file with matching flow count)
            
        Returns:
            features_list: list of feature arrays
            labels_list: list of labels
            filenames_list: list of filenames
        """
        pcap_files = glob.glob(os.path.join(directory, "*.pcap"))
        pcap_files.extend(glob.glob(os.path.join(directory, "*.pcapng")))
        
        if len(pcap_files) == 0:
            logger.warning(f"No PCAP files found in {directory}")
            return [], [], []
        
        # If max_flows is None, use all files
        if max_flows is None:
            # Use all PCAP files in the directory
            selected_files = sorted(pcap_files)
            logger.info(f"Reading all PCAP files from {directory} (flow counts will be counted during processing)")
        else:
            # Select file(s) based on required flow count (backward compatibility)
            # Extract flow counts from filenames and create mapping
            file_flow_map = {}
            for pcap_file in pcap_files:
                flow_count = extract_flow_count(os.path.basename(pcap_file))
                if flow_count is not None:
                    file_flow_map[flow_count] = pcap_file
            
            if len(file_flow_map) == 0:
                logger.warning(f"No flow count found in filenames in {directory}, using all files")
                selected_files = sorted(pcap_files)
            else:
                # Try to find exact match first
                if max_flows in file_flow_map:
                    selected_files = [file_flow_map[max_flows]]
                    logger.info(f"Found exact match: file with {max_flows} flows")
                else:
                    # Find closest match
                    available_flows = sorted(file_flow_map.keys())
                    closest_flow = min(available_flows, key=lambda x: abs(x - max_flows))
                    selected_files = [file_flow_map[closest_flow]]
                    logger.info(f"Using closest match: file with {closest_flow} flows (requested: {max_flows})")
        
        features_list = []
        labels_list = []
        filenames_list = []
        
        logger.info(f"Selected {len(selected_files)} PCAP file(s) from {directory} based on flow count")
        
        total_flows_extracted = 0
        
        for pcap_path in selected_files:
            filename = os.path.basename(pcap_path)
            expected_flow_count = extract_flow_count(filename)
            logger.info(f"开始解析 {filename} (预期流数: {expected_flow_count})...")
            logger.info(f"  步骤1: 识别流（通过5元组: IP+端口+协议，包间隔>120秒则开始新流）")
            logger.info(f"  步骤2: 为每个流提取4维特征（每个数据包，最多处理前5000个包）")
            logger.info(f"  步骤3: 序列对齐到1024长度")
            logger.info("")
            
            # Parse file and extract all flows (with 120s timeout)
            flow_features_list = self.parse_pcap_file(pcap_path, extract_flows=True, flow_timeout=120.0)
            
            if flow_features_list is None:
                logger.warning(f"  ✗ Failed to parse {filename}")
                continue
            
            if isinstance(flow_features_list, list):
                # Multiple flows extracted
                extracted_count = 0
                for flow_features in flow_features_list:
                    if flow_features is not None:
                        features_list.append(flow_features)
                        labels_list.append(label)
                        filenames_list.append(filename)
                        total_flows_extracted += 1
                        extracted_count += 1
                logger.info(f"  ✓ {filename} 解析完成，共提取 {extracted_count} 条流")
            else:
                # Single flow (fallback - should not happen with extract_flows=True)
                if flow_features_list is not None:
                    features_list.append(flow_features_list)
                    labels_list.append(label)
                    filenames_list.append(filename)
                    total_flows_extracted += 1
                    logger.info(f"  ✓ {filename} 解析完成，提取 1 条流")
                else:
                    logger.warning(f"  ✗ {filename} 解析失败")
        
        logger.info(f"从 {directory} 成功提取 {total_flows_extracted} 条流 (来自 {len(selected_files)} 个文件)")
        
        return features_list, labels_list, filenames_list


def load_dataset(benign_dir, malicious_dir, sequence_length=1024):
    """
    Load dataset from benign and malicious directories
    
    Reads all PCAP files from the specified directories.
    Flow counts are counted during processing.
    
    Args:
        benign_dir: Directory containing benign PCAP files
        malicious_dir: Directory containing malicious PCAP files
        sequence_length: Sequence length for feature extraction
        
    Returns:
        X: numpy array of shape (N, L, 4)
        y: numpy array of shape (N,)
        filenames: list of filenames
    """
    parser = PCAPParser(sequence_length=sequence_length)
    
    # Load benign samples - read all pcap files
    logger.info(f"Loading benign samples from {benign_dir}...")
    benign_features, benign_labels, benign_files = parser.parse_directory(
        benign_dir, label=0, max_flows=None
    )
    
    # Load malicious samples - read all pcap files
    logger.info(f"Loading malicious samples from {malicious_dir}...")
    malicious_features, malicious_labels, malicious_files = parser.parse_directory(
        malicious_dir, label=1, max_flows=None
    )
    
    # Combine
    all_features = benign_features + malicious_features
    all_labels = benign_labels + malicious_labels
    all_files = benign_files + malicious_files
    
    if len(all_features) == 0:
        logger.error("No valid samples loaded!")
        return None, None, None
    
    # Convert to numpy arrays
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    n_benign = np.sum(y==0)
    n_malicious = np.sum(y==1)
    logger.info(f"Dataset loaded: {X.shape[0]} samples total")
    logger.info(f"  Benign flows: {n_benign}")
    logger.info(f"  Malicious flows: {n_malicious}")
    
    return X, y, all_files
