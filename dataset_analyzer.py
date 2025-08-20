import os
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from torchvision import transforms

class UltrasoundDatasetAnalyzer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.categories = {
            "CN_ON": "블라인드존 없음 + 조직 없음 (배경)",
            "CN_OY": "블라인드존 없음 + 조직 있음 (Ground Truth)",
            "CY_ON": "블라인드존 있음 + 조직 없음 (구조적 잡음)",
            "CY_OY": "블라인드존 있음 + 조직 있음 (복원 대상)"
        }
        self.file_stats = defaultdict(list)
        
    def analyze_dataset(self):
        """Analyze the dataset structure and statistics"""
        print("=== 초음파 데이터셋 분석 ===")
        print(f"데이터 경로: {self.data_path}")
        
        # Collect file statistics
        for image_file in self.data_path.glob("*.bmp"):
            filename = image_file.name
            
            # Determine category
            category = None
            for cat_key in self.categories.keys():
                if filename.startswith(cat_key):
                    category = cat_key
                    break
            
            if category:
                self.file_stats[category].append(image_file)
        
        # Print statistics
        total_files = 0
        for category, files in self.file_stats.items():
            count = len(files)
            total_files += count
            print(f"{category} ({self.categories[category]}): {count}개 파일")
        
        print(f"\n총 파일 수: {total_files}개")
        
        return self.file_stats
    
    def analyze_image_properties(self, sample_count=5):
        """Analyze image properties for each category"""
        print("\n=== 이미지 속성 분석 ===")
        
        for category, files in self.file_stats.items():
            if not files:
                continue
                
            print(f"\n{category} 카테고리 분석:")
            
            # Sample images for analysis
            sample_files = files[:min(sample_count, len(files))]
            sizes = []
            intensities = []
            
            for file_path in sample_files:
                img = Image.open(file_path).convert('L')
                img_array = np.array(img)
                
                sizes.append(img.size)
                intensities.append({
                    'mean': np.mean(img_array),
                    'std': np.std(img_array),
                    'min': np.min(img_array),
                    'max': np.max(img_array)
                })
            
            # Print size statistics
            if sizes:
                unique_sizes = list(set(sizes))
                print(f"  이미지 크기: {unique_sizes}")
            
            # Print intensity statistics
            if intensities:
                mean_intensity = np.mean([i['mean'] for i in intensities])
                mean_std = np.mean([i['std'] for i in intensities])
                print(f"  평균 픽셀 강도: {mean_intensity:.2f}")
                print(f"  픽셀 강도 표준편차: {mean_std:.2f}")
    
    def create_training_recommendations(self):
        """Create DDPM training recommendations based on analysis"""
        print("\n=== DDPM 훈련 데이터셋 구성 권장사항 ===")
        
        cn_on_count = len(self.file_stats.get("CN_ON", []))
        cn_oy_count = len(self.file_stats.get("CN_OY", []))
        cy_on_count = len(self.file_stats.get("CY_ON", []))
        cy_oy_count = len(self.file_stats.get("CY_OY", []))
        
        print("\n1. DDPM 사전 훈련용 데이터:")
        print(f"   - CN_OY (Ground Truth): {cn_oy_count}개")
        print(f"   - CN_ON (배경 신호): {cn_on_count}개")
        print(f"   - 총 훈련 데이터: {cn_on_count + cn_oy_count}개")
        
        print("\n2. DDRM 손상 모델 추정용:")
        print(f"   - CY_ON (블라인드존 패턴): {cy_on_count}개")
        print(f"   - CY_OY (테스트 대상): {cy_oy_count}개")
        
        print("\n3. 데이터 균형 분석:")
        total_clean = cn_on_count + cn_oy_count
        total_corrupted = cy_on_count + cy_oy_count
        print(f"   - 깨끗한 데이터: {total_clean}개")
        print(f"   - 손상된 데이터: {total_corrupted}개")
        print(f"   - 비율: {total_clean/total_corrupted:.2f}:1")
        
        if total_clean < 100:
            print("\n⚠️  경고: 훈련 데이터가 부족할 수 있습니다. 데이터 증강을 고려하세요.")
        
        return {
            "training_data": ["CN_ON", "CN_OY"],
            "training_count": total_clean,
            "degradation_estimation": ["CY_ON", "CN_ON"],
            "test_data": ["CY_OY"]
        }
    
    def estimate_noise_pattern(self, sample_count=10):
        """Estimate blind zone noise pattern for DDRM"""
        print("\n=== 블라인드존 노이즈 패턴 추정 ===")
        
        cn_on_files = self.file_stats.get("CN_ON", [])
        cy_on_files = self.file_stats.get("CY_ON", [])
        
        if not cn_on_files or not cy_on_files:
            print("CN_ON 또는 CY_ON 데이터가 없어 노이즈 패턴 추정이 불가능합니다.")
            return None
        
        # Sample images
        cn_samples = cn_on_files[:min(sample_count, len(cn_on_files))]
        cy_samples = cy_on_files[:min(sample_count, len(cy_on_files))]
        
        noise_patterns = []
        
        for cn_file, cy_file in zip(cn_samples, cy_samples):
            cn_img = np.array(Image.open(cn_file).convert('L'))
            cy_img = np.array(Image.open(cy_file).convert('L'))
            
            # Estimate noise as difference
            noise_pattern = cy_img.astype(float) - cn_img.astype(float)
            noise_patterns.append(noise_pattern)
        
        # Calculate average noise pattern
        avg_noise = np.mean(noise_patterns, axis=0)
        noise_std = np.std(noise_patterns, axis=0)
        
        print(f"블라인드존 노이즈 통계:")
        print(f"  평균 노이즈 강도: {np.mean(avg_noise):.2f}")
        print(f"  노이즈 표준편차: {np.mean(noise_std):.2f}")
        print(f"  노이즈 범위: [{np.min(avg_noise):.2f}, {np.max(avg_noise):.2f}]")
        
        return avg_noise, noise_std

def main():
    # Analyze the dataset
    data_path = "/home/ubuntu/Desktop/JY/ultrasound_inp/diffusers/data/train_CN_CY_ALL"
    
    analyzer = UltrasoundDatasetAnalyzer(data_path)
    
    # Run analysis
    file_stats = analyzer.analyze_dataset()
    analyzer.analyze_image_properties()
    recommendations = analyzer.create_training_recommendations()
    noise_stats = analyzer.estimate_noise_pattern()
    
    print("\n=== 분석 완료 ===")
    print("DDPM 훈련을 시작할 준비가 완료되었습니다.")
    
if __name__ == "__main__":
    main()