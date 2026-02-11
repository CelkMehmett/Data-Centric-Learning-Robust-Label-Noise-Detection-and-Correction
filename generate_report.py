import json
import glob
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def get_base64_image(plt_figure):
    """Matplotlib figürünü base64 string'e çevirir"""
    buf = BytesIO()
    plt_figure.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return image_base64

def generate_report():
    print("HTML Raporu Oluşturuluyor...")
    
    # JSON sonuç dosyalarını bul
    result_files = glob.glob("pipeline_results_*.json")
    
    html_content = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data-Centric AI Proje Raporu</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; }
            .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 20px rgba(0,0,0,0.1); border-radius: 8px; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            .card { background: #fff; border: 1px solid #ddd; border-radius: 4px; padding: 20px; margin-bottom: 20px; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #3498db; color: white; }
            tr:hover { background-color: #f5f5f5; }
            .metric-box { display: inline-block; background: #ecf0f1; padding: 15px; border-radius: 4px; margin-right: 15px; text-align: center; min-width: 120px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
            .metric-label { font-size: 14px; color: #7f8c8d; }
            .footer { margin-top: 50px; text-align: center; color: #95a5a6; font-size: 12px; }
            .status-badge { padding: 5px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }
            .status-success { background-color: #d4edda; color: #155724; }
            .status-warning { background-color: #fff3cd; color: #856404; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Data-Centric AI Proje Sonuçları</h1>
            <p>Bu rapor, CIFAR-10 veriseti üzerindeki gürültü tespiti ve temizleme deneylerinin sonuçlarını özetler.</p>
    """
    
    if not result_files:
        html_content += """
            <div class="card">
                <h3>Henüz Sonuç Bulunamadı</h3>
                <p>Lütfen önce <code>python3 run_pipeline.py</code> komutunu çalıştırarak deneyleri tamamlayın.</p>
            </div>
        """
    else:
        for file in result_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                # Dosya adından gürültü tipini çıkar (örn: pipeline_results_symmetric_0.2.json)
                filename_parts = file.replace("pipeline_results_", "").replace(".json", "").split("_")
                noise_type = filename_parts[0]
                noise_rate = filename_parts[1]
                
                # Verileri Çıkar
                baseline_acc = data.get('baseline', {}).get('acc', 0)
                baseline_f1 = data.get('baseline', {}).get('f1', 0)
                
                num_issues = data.get('detection', {}).get('num_issues', 0)
                precision = data.get('detection', {}).get('precision', 0)
                recall = data.get('detection', {}).get('recall', 0)
                
                html_content += f"""
                <div class="card">
                    <h2>Deney: {noise_type.capitalize()} Gürültü (Oran: {noise_rate})</h2>
                    
                    <h3>1. Baz Model (Baseline) Performansı</h3>
                    <div>
                        <div class="metric-box">
                            <div class="metric-value">%{baseline_acc*100:.1f}</div>
                            <div class="metric-label">Doğruluk (Accuracy)</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{baseline_f1:.3f}</div>
                            <div class="metric-label">F1 Skoru</div>
                        </div>
                    </div>
                    
                    <h3>2. Gürültü Tespit Performansı (Cleanlab)</h3>
                    <p>Tespit Edilen Etiket Hataları: <strong>{num_issues}</strong></p>
                    <table>
                        <thead>
                            <tr>
                                <th>Metrik</th>
                                <th>Değer</th>
                                <th>Açıklama</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Kesinlik (Precision)</td>
                                <td><strong>{precision:.3f}</strong></td>
                                <td>Tespit edilenlerin ne kadarı gerçekten gürültülüydü?</td>
                            </tr>
                            <tr>
                                <td>Duyarlılık (Recall)</td>
                                <td><strong>{recall:.3f}</strong></td>
                                <td>Gerçek gürültülerin ne kadarını yakaladık?</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """
            except Exception as e:
                html_content += f"<p style='color:red'>Hata: {file} dosyası okunurken sorun oluştu: {str(e)}</p>"

    # Grafik Ekleme (Opsiyonel - Eğer sonuçlar varsa)
    # Burada karşılaştırmalı bir bar grafiği oluşturabiliriz
    if result_files:
        html_content += "<h2>📈 Genel Performans Karşılaştırması</h2>"
        # Basit bir bar plot
        # data toplama
        labels = []
        precs = []
        reco = []
        
        for file in result_files:
            try:
                with open(file, 'r') as f:
                    d = json.load(f)
                fname = file.replace("pipeline_results_", "").replace(".json", "")
                labels.append(fname)
                precs.append(d.get('detection', {}).get('precision', 0))
                reco.append(d.get('detection', {}).get('recall', 0))
            except: pass
            
        if labels:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = range(len(labels))
            width = 0.35
            ax.bar([i - width/2 for i in x], precs, width, label='Kesinlik (Precision)', color='#3498db')
            ax.bar([i + width/2 for i in x], reco, width, label='Duyarlılık (Recall)', color='#2ecc71')
            ax.set_ylabel('Skor')
            ax.set_title('Gürültü Tespit Başarısı')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            
            img_b64 = get_base64_image(fig)
            html_content += f'<img src="data:image/png;base64,{img_b64}" style="width:100%; max-width:800px; display:block; margin:auto;">'

    html_content += """
            <div class="footer">
                <p>Oluşturulma Tarihi: otomatik • Data-Centric AI Pipeline</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open("report.html", "w") as f:
        f.write(html_content)
    
    print(f"Rapor başarıyla oluşturuldu: {os.path.abspath('report.html')}")

if __name__ == "__main__":
    generate_report()
