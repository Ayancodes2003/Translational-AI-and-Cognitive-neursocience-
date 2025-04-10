"""
Clinical Report Generator

This module generates clinical reports based on model predictions and insights.
"""

import os
import numpy as np
import pandas as pd
import torch
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinicalReportGenerator:
    """Class for generating clinical reports based on model predictions and insights."""
    
    def __init__(self, risk_assessor=None, contribution_analyzer=None, output_dir=None):
        """
        Initialize the clinical report generator.
        
        Args:
            risk_assessor (RiskAssessor, optional): Risk assessor
            contribution_analyzer (ModalityContributionAnalyzer, optional): Modality contribution analyzer
            output_dir (str, optional): Directory to save reports
        """
        self.risk_assessor = risk_assessor
        self.contribution_analyzer = contribution_analyzer
        self.output_dir = output_dir or 'results/clinical_reports'
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_report(self, sample_data, sample_label=None, sample_id=None):
        """
        Generate a clinical report for a single sample.
        
        Args:
            sample_data (dict): Sample data for each modality
            sample_label (torch.Tensor, optional): Sample label
            sample_id (int, optional): Sample ID
        
        Returns:
            dict: Clinical report
        """
        logger.info(f"Generating clinical report for sample {sample_id}")
        
        # Create report dictionary
        report = {
            'sample_id': sample_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add true label if available
        if sample_label is not None:
            report['true_label'] = int(sample_label.cpu().numpy()[0, 0])
            if sample_label.shape[1] > 1:
                report['phq8_score'] = float(sample_label.cpu().numpy()[0, 1])
        
        # Get model prediction
        if isinstance(sample_data, dict):
            sample_data = {k: v.to(self.risk_assessor.device) for k, v in sample_data.items()}
        else:
            sample_data = sample_data.to(self.risk_assessor.device)
        
        with torch.no_grad():
            output = self.risk_assessor.model(sample_data)
            probability = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        report['depression_probability'] = float(probability)
        
        # Get risk assessment
        if self.risk_assessor is not None:
            if 'phq8_score' in report:
                risk_report = self.risk_assessor.generate_risk_report(probability, report['phq8_score'])
            else:
                risk_report = self.risk_assessor.generate_risk_report(probability)
            
            report['risk_level'] = risk_report['risk_level']
            report['suggestions'] = risk_report['suggestions']
            
            if 'phq8_interpretation' in risk_report:
                report['phq8_interpretation'] = risk_report['phq8_interpretation']
        
        # Get modality contributions
        if self.contribution_analyzer is not None:
            contribution_report = self.contribution_analyzer.generate_contribution_report(sample_data, sample_label)
            report['modality_contributions'] = contribution_report['contributions']
        
        # Generate observations
        report['observations'] = self._generate_observations(report)
        
        return report
    
    def _generate_observations(self, report):
        """
        Generate observations based on the report.
        
        Args:
            report (dict): Clinical report
        
        Returns:
            list: Observations
        """
        observations = []
        
        # Add observation about depression probability
        prob = report['depression_probability']
        if prob < 0.3:
            observations.append("Low probability of depression detected.")
        elif prob < 0.7:
            observations.append("Moderate probability of depression detected.")
        else:
            observations.append("High probability of depression detected.")
        
        # Add observation about PHQ-8 score if available
        if 'phq8_score' in report:
            score = report['phq8_score']
            if score < 5:
                observations.append("PHQ-8 score indicates minimal or no depression.")
            elif score < 10:
                observations.append("PHQ-8 score indicates mild depression.")
            elif score < 15:
                observations.append("PHQ-8 score indicates moderate depression.")
            elif score < 20:
                observations.append("PHQ-8 score indicates moderately severe depression.")
            else:
                observations.append("PHQ-8 score indicates severe depression.")
        
        # Add observation about modality contributions if available
        if 'modality_contributions' in report:
            contributions = report['modality_contributions']
            
            # Find the modality with the highest contribution
            max_modality = max(contributions, key=contributions.get)
            max_contribution = contributions[max_modality]
            
            if max_contribution > 0.5:
                observations.append(f"The {max_modality} modality is the primary indicator, contributing {max_contribution:.1%} to the assessment.")
            
            # Add observations about specific modalities
            for modality, contribution in contributions.items():
                if modality == 'eeg' and contribution > 0.3:
                    observations.append("EEG patterns show significant indicators of altered brain activity.")
                elif modality == 'audio' and contribution > 0.3:
                    observations.append("Speech patterns show notable changes in vocal characteristics.")
                elif modality == 'text' and contribution > 0.3:
                    observations.append("Linguistic patterns show significant indicators in language use.")
        
        # Add observation about risk level
        if 'risk_level' in report:
            risk_level = report['risk_level']
            if risk_level == 'Low':
                observations.append("Overall risk assessment indicates low risk for depression.")
            elif risk_level == 'Moderate':
                observations.append("Overall risk assessment indicates moderate risk for depression. Regular monitoring is recommended.")
            else:
                observations.append("Overall risk assessment indicates high risk for depression. Professional intervention is recommended.")
        
        return observations
    
    def save_report(self, report, path=None):
        """
        Save a clinical report to a file.
        
        Args:
            report (dict): Clinical report
            path (str, optional): Path to save the report
        """
        if path is None:
            sample_id = report.get('sample_id', 'unknown')
            path = os.path.join(self.output_dir, f'clinical_report_{sample_id}.json')
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Saved clinical report to {path}")
    
    def generate_pdf_report(self, report, path=None):
        """
        Generate a PDF version of the clinical report.
        
        Args:
            report (dict): Clinical report
            path (str, optional): Path to save the PDF report
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        except ImportError:
            logger.error("reportlab is not installed. Cannot generate PDF report.")
            return
        
        if path is None:
            sample_id = report.get('sample_id', 'unknown')
            path = os.path.join(self.output_dir, f'clinical_report_{sample_id}.pdf')
        
        logger.info(f"Generating PDF report to {path}")
        
        # Create PDF document
        doc = SimpleDocTemplate(path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles
        styles.add(ParagraphStyle(name='Title', parent=styles['Heading1'], alignment=1))
        styles.add(ParagraphStyle(name='Subtitle', parent=styles['Heading2'], alignment=1))
        styles.add(ParagraphStyle(name='Section', parent=styles['Heading2']))
        styles.add(ParagraphStyle(name='Subsection', parent=styles['Heading3']))
        
        # Create content
        content = []
        
        # Title
        content.append(Paragraph("Mental Health Assessment Report", styles['Title']))
        content.append(Spacer(1, 12))
        
        # Timestamp
        content.append(Paragraph(f"Generated on: {report['timestamp']}", styles['Subtitle']))
        content.append(Spacer(1, 24))
        
        # Sample ID
        if 'sample_id' in report:
            content.append(Paragraph(f"Sample ID: {report['sample_id']}", styles['Normal']))
            content.append(Spacer(1, 12))
        
        # Depression probability
        content.append(Paragraph("Depression Assessment", styles['Section']))
        content.append(Spacer(1, 6))
        content.append(Paragraph(f"Depression Probability: {report['depression_probability']:.1%}", styles['Normal']))
        content.append(Spacer(1, 12))
        
        # PHQ-8 score if available
        if 'phq8_score' in report:
            content.append(Paragraph(f"PHQ-8 Score: {report['phq8_score']}", styles['Normal']))
            if 'phq8_interpretation' in report:
                content.append(Paragraph(f"Interpretation: {report['phq8_interpretation']}", styles['Normal']))
            content.append(Spacer(1, 12))
        
        # Risk level
        if 'risk_level' in report:
            content.append(Paragraph("Risk Assessment", styles['Section']))
            content.append(Spacer(1, 6))
            content.append(Paragraph(f"Risk Level: {report['risk_level']}", styles['Normal']))
            content.append(Spacer(1, 12))
        
        # Modality contributions
        if 'modality_contributions' in report:
            content.append(Paragraph("Modality Contributions", styles['Section']))
            content.append(Spacer(1, 6))
            
            # Create table for modality contributions
            contributions = report['modality_contributions']
            data = [['Modality', 'Contribution']]
            for modality, contribution in contributions.items():
                data.append([modality.capitalize(), f"{contribution:.1%}"])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(table)
            content.append(Spacer(1, 12))
        
        # Observations
        if 'observations' in report:
            content.append(Paragraph("Clinical Observations", styles['Section']))
            content.append(Spacer(1, 6))
            
            for observation in report['observations']:
                content.append(Paragraph(f"• {observation}", styles['Normal']))
                content.append(Spacer(1, 6))
            
            content.append(Spacer(1, 12))
        
        # Suggestions
        if 'suggestions' in report:
            content.append(Paragraph("Recommendations", styles['Section']))
            content.append(Spacer(1, 6))
            
            for suggestion in report['suggestions']:
                content.append(Paragraph(f"• {suggestion}", styles['Normal']))
                content.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(content)
        
        logger.info(f"Generated PDF report at {path}")
    
    def batch_generate_reports(self, test_loader, num_samples=10):
        """
        Generate clinical reports for multiple samples.
        
        Args:
            test_loader (DataLoader): Test data loader
            num_samples (int): Number of samples to generate reports for
        
        Returns:
            list: Clinical reports
        """
        logger.info(f"Generating clinical reports for {num_samples} samples")
        
        reports = []
        count = 0
        
        for data, target in test_loader:
            # Generate reports for each sample in the batch
            for i in range(len(target)):
                if count >= num_samples:
                    break
                
                # Extract single sample
                if isinstance(data, dict):
                    sample_data = {k: v[i:i+1] for k, v in data.items()}
                else:
                    sample_data = data[i:i+1]
                
                sample_target = target[i:i+1]
                
                # Generate report
                report = self.generate_report(sample_data, sample_target, sample_id=count)
                reports.append(report)
                
                # Save report
                self.save_report(report)
                
                # Generate PDF report
                self.generate_pdf_report(report)
                
                count += 1
            
            if count >= num_samples:
                break
        
        # Save all reports to a single file
        with open(os.path.join(self.output_dir, 'all_clinical_reports.json'), 'w') as f:
            json.dump(reports, f, indent=4)
        
        return reports


def main():
    """Main function."""
    import argparse
    import torch
    from torch.utils.data import DataLoader
    import pickle
    import sys
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from clinical_insights.risk_assessment import RiskAssessor
    from clinical_insights.modality_contribution import ModalityContributionAnalyzer
    
    parser = argparse.ArgumentParser(description='Clinical report generation')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/fusion/processed',
                        help='Path to the processed fusion data')
    
    # Report arguments
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate reports for')
    parser.add_argument('--output_dir', type=str, default='results/clinical_reports',
                        help='Directory to save reports')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # TODO: Create model based on checkpoint and load state dict
    
    # Load test data
    with open(os.path.join(args.data_path, 'fusion_dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    
    # Create test loader
    # TODO: Create test loader based on dataset
    
    # Create risk assessor
    risk_assessor = RiskAssessor(
        threshold_low=0.3,
        threshold_high=0.7,
        output_dir=os.path.join(args.output_dir, 'risk_assessment')
    )
    risk_assessor.model = model
    risk_assessor.device = device
    
    # Create modality contribution analyzer
    contribution_analyzer = ModalityContributionAnalyzer(
        model=model,
        device=device,
        output_dir=os.path.join(args.output_dir, 'modality_contribution')
    )
    
    # Create clinical report generator
    report_generator = ClinicalReportGenerator(
        risk_assessor=risk_assessor,
        contribution_analyzer=contribution_analyzer,
        output_dir=args.output_dir
    )
    
    # Generate reports
    reports = report_generator.batch_generate_reports(test_loader, args.num_samples)


if __name__ == '__main__':
    main()
