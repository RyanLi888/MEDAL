"""
PCAP Parser Module
Extracts 5-dimensional features from PCAP files
"""
import numpy as np
import os
import glob
import re
from scapy.all import rdpcap, IP, TCP
from scapy.error import Scapy_Exception
import logging

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
    
    def __init__(self, sequence_length=1024):
        """
        Args:
            sequence_length: Maximum number of packets to extract (L=1024)
        """
        self.sequence_length = sequence_length
        
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
    
    def parse_pcap_file(self, pcap_path, extract_flows=True):
        """
        Parse a single PCAP file and extract 5D features
        
        If extract_flows=True, extracts multiple flows from the PCAP file.
        Each flow is identified by 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol).
        
        Args:
            pcap_path: Path to PCAP file
            extract_flows: If True, extract multiple flows; if False, treat entire file as one flow
            
        Returns:
            If extract_flows=True: list of feature arrays (one per flow)
            If extract_flows=False: single feature array of shape (L, 5)
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
        
        # Group packets by flow (5-tuple)
        flows = {}
        for pkt in packets:
            flow_key = self._extract_flow_key(pkt)
            if flow_key is None:
                continue
            
            if flow_key not in flows:
                flows[flow_key] = []
            flows[flow_key].append(pkt)
        
        if len(flows) == 0:
            logger.warning(f"No valid flows found in {pcap_path}")
            return []
        
        logger.info(f"在 {os.path.basename(pcap_path)} 中发现 {len(flows)} 条流，开始提取5维特征...")
        logger.info(f"  每个数据包提取: [Length, Log-IAT, Direction, Flags, Window]")
        logger.info(f"  每个流转换为: 1024×5 的特征序列")
        logger.info("")
        
        # Extract features for each flow
        all_flow_features = []
        flow_count = 0
        
        for flow_key, flow_packets in flows.items():
            flow_count += 1
            flow_features = self._parse_single_flow(flow_packets)
            
            if flow_features is not None:
                all_flow_features.append(flow_features)
                # 每10条流或最后一条流输出详细日志
                if flow_count % 10 == 0 or flow_count == len(flows):
                    logger.info(f"  ✓ 完成 {flow_count}/{len(flows)} 条流 (当前流包含 {len(flow_packets)} 个数据包，已提取 {flow_features.shape[0]}×5 特征)")
            else:
                logger.warning(f"  ✗ 流 {flow_count}/{len(flows)} 解析失败")
        
        logger.info("")
        logger.info(f"✓ 成功提取 {len(all_flow_features)}/{len(flows)} 条流的5维特征序列")
        
        return all_flow_features if len(all_flow_features) > 0 else []
    
    def _parse_single_flow(self, packets):
        """
        Parse a single flow (list of packets) and extract 5D features
        
        Args:
            packets: List of Scapy packets belonging to one flow
            
        Returns:
            features: numpy array of shape (L, 5)
                     [Length, Log-IAT, Direction, Flags, Window]
        """
        if len(packets) == 0:
            return None
        
        # Sort packets by time
        packets = sorted(packets, key=lambda p: float(p.time))
        
        # Extract features
        features_list = []
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
            length_norm = min(length / 1500.0, 1.0)
            
            # Feature 1: Log-IAT (Inter-Arrival Time)
            if prev_time is None:
                log_iat = 0.0
                prev_time = float(pkt.time)
            else:
                iat = float(pkt.time) - prev_time
                log_iat = np.log(iat + 1e-7)
                prev_time = float(pkt.time)
            
            # Feature 2: Direction (+1: client->server, -1: server->client)
            if pkt[IP].src == client_ip:
                direction = 1.0
            else:
                direction = -1.0
            
            # Feature 3: Flags (TCP flags encoded)
            flags_val = 0.0
            if TCP in pkt:
                tcp_flags = pkt[TCP].flags
                syn = 1.0 if 'S' in str(tcp_flags) else 0.0
                fin = 1.0 if 'F' in str(tcp_flags) else 0.0
                rst = 1.0 if 'R' in str(tcp_flags) else 0.0
                psh = 1.0 if 'P' in str(tcp_flags) else 0.0
                ack = 1.0 if 'A' in str(tcp_flags) else 0.0
                
                flags_val = (syn * 16 + fin * 8 + rst * 4 + psh * 2 + ack) / 31.0
            
            # Feature 4: Window (TCP window size normalized)
            window = 0.0
            if TCP in pkt:
                window = min(pkt[TCP].window / 65535.0, 1.0)
            
            features_list.append([length_norm, log_iat, direction, flags_val, window])
        
        if len(features_list) == 0:
            return None
        
        # Convert to numpy array
        features = np.array(features_list, dtype=np.float32)
        
        # Sequence alignment (truncate or pad to L=1024)
        features = self._align_sequence(features)
        
        return features
    
    def _align_sequence(self, features):
        """
        Align sequence to fixed length L
        
        Args:
            features: numpy array of shape (N, 5)
            
        Returns:
            aligned_features: numpy array of shape (L, 5)
        """
        n_packets = features.shape[0]
        
        if n_packets >= self.sequence_length:
            # Truncate: keep first L packets
            return features[:self.sequence_length]
        else:
            # Pad: append zeros
            padding = np.zeros((self.sequence_length - n_packets, 5), dtype=np.float32)
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
        
        # Extract flow counts from filenames and create mapping
        file_flow_map = {}
        for pcap_file in pcap_files:
            flow_count = extract_flow_count(os.path.basename(pcap_file))
            if flow_count is not None:
                file_flow_map[flow_count] = pcap_file
        
        if len(file_flow_map) == 0:
            logger.warning(f"No flow count found in filenames in {directory}, using all files")
            # Use all files if no flow count found
            selected_files = sorted(pcap_files)
        else:
            # If max_flows is None, use all files
            if max_flows is None:
                # Use all files
                selected_files = sorted(file_flow_map.values())
                logger.info(f"Reading all PCAP files from {directory} (flow counts will be counted during processing)")
            else:
                # Select file(s) based on required flow count (backward compatibility)
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
            logger.info(f"  步骤1: 识别流（通过5元组: IP+端口+协议）")
            logger.info(f"  步骤2: 为每个流提取5维特征（每个数据包）")
            logger.info(f"  步骤3: 序列对齐到1024长度")
            logger.info("")
            
            # Parse file and extract all flows
            flow_features_list = self.parse_pcap_file(pcap_path, extract_flows=True)
            
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
        X: numpy array of shape (N, L, 5)
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
