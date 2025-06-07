import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.formula.api as smf
from scipy import stats

from django.conf import settings
from django.core.management.base import BaseCommand

from din.models import Test, Response

COLORS = ['#0072B2', '#CC79A7', '#009E73']
LABELS = ["Unprocessed", "NH Vocoded", "EH Vocoded"]

def tost_from_summary(mean1, std1, nobs1, mean2, std2, nobs2, 
                      delta=1, alpha=0.05, equal_var=False):
    # Lower bound test: mean1 > mean2 - delta
    t1, p1_two_sided = stats.ttest_ind_from_stats(mean1, std1, nobs1,
                                            mean2 + delta, std2, nobs2,
                                            equal_var=equal_var)
    p1 = p1_two_sided / 2  # one-sided

    # Upper bound test: mean1 < mean2 + delta
    t2, p2_two_sided = stats.ttest_ind_from_stats(mean1, std1, nobs1,
                                            mean2 - delta, std2, nobs2,
                                            equal_var=equal_var)
    p2 = p2_two_sided / 2  # one-sided

    equivalent = p1 < alpha and p2 < alpha
    return {
        "p1": p1,
        "p2": p2,
        "max_p": max(p1, p2),
        "equivalent": equivalent
    }

class Command(BaseCommand):
    help = "Describe what this command does"

    def make_boxplot(self, result: pd.DataFrame):
        groups = result['test'].unique()
        plot_data = [result[result['test'] == g]['srt'] for g in groups]

        fig, ax = plt.subplots(figsize=(8, 4))

        parts = ax.violinplot(
            plot_data,
            showextrema=False
        )
        
        box = ax.boxplot(
            plot_data,
            patch_artist=True,
            tick_labels=LABELS
        )
        
        for whisker in box['whiskers']:
            whisker.set(color='black', linewidth=2)
        for cap in box['caps']:
            cap.set(color='black', linewidth=2)
        for median in box['medians']:
            median.set(color='black', linewidth=2)
            
        for flier in box['fliers']:
            flier.set(marker='o', color='red', alpha=0.5)
            
        for patch, color, pc in zip(box['boxes'], COLORS, parts['bodies']):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.4)        # Transparency
            pc.set_linewidth(1.2)
            
        d = .15
        for i, x in enumerate(plot_data, 1):
            mean = np.median(x)
            ax.scatter(i, mean, s=30, color='black', zorder=3)
            ax.plot([i, i +d], [mean, mean], ls="dashdot", color="black", zorder=3)
            ax.text(
                i + d,
                mean,
                f"{mean:.2f} dB",
                fontsize=14,
                va="center",
                bbox = dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="round",
                    pad=0.15
                ),
                zorder=10 # to make sure the line is on top
            )
        
            
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel("SRT [dB]", fontsize=15)
        plt.tight_layout()
        plt.savefig(settings.MEDIA_ROOT / "boxplot_din.png", dpi=600)        

    def make_age_scatter_plot(self, result: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(7, 4))
        
        for (label, group), color, plabel in zip(result.groupby("test"), COLORS, LABELS):
            z = np.polyfit(group['age'], group['srt'], 1)
            p = np.poly1d(z)
            print(z)
            x = np.arange(group['age'].min()*1.1, group['age'].max() * 0.99)
            ax.plot(x, p(x), color=color, alpha=.5, linestyle='dashed', linewidth=2)
            ax.scatter(group['age'], group['srt'], color=color, label=plabel, alpha=.6)

        for (pk, group) in result.groupby("pk"):
            age = group.iloc[0].age
            ax.plot([age, age], [group.srt.min(), group.srt.max()], 
                color='grey', alpha=.5, zorder=-10)

        ax.legend(ncol=3, fontsize=12)     
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylabel("SRT [dB]", fontsize=15)
        ax.set_xlabel("Age", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        plt.savefig(settings.MEDIA_ROOT / "ageplot_din.png", dpi=600)    

        
        model = smf.ols('srt ~ age', data=result).fit()
        print(model.summary())
    
    def make_comparison_plot(self, result: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(7, 4))
        cmap = plt.get_cmap("nipy_spectral")
        norm = mpl.colors.Normalize(vmin=result.pk.min(), vmax=result.pk.max())


        for (pk, group) in result.groupby("pk"):
            color = cmap(norm(pk))
            ax.plot(group.test, group.srt, color='grey', alpha=.3, zorder=-1)
            ax.scatter(group.test, group.srt, color='black')

        ax.set_xticks(range(0, 3), LABELS)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylabel("SRT [dB]", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        plt.savefig(settings.MEDIA_ROOT / "comparison_plot.png", dpi=600)    

    def vs_studies(self, result: pd.DataFrame):
        
        nh_data = np.array([
            (-8.8, 0.6, 23),  # Smits 2013
            (-9.3, 0.7, 16),  # Smits 2016
            (-9.5, 1.0, 12),  # de Graaf est fig 1.
            (-9.3, 0.7, 12),  # Kaandorp 2015
            (-8.4, 0.6, 18),  # Stronks 2025
        ])

        ci_data = np.array([
            (-3.6, 1.7, 16),  # de Graaf est fig 1.
            (-1.8, 2.7, 24),  # Kaandorp 2015
            (-1.5, 2.5, 18),  # Stronks 2025
            (-1.4, 3.8, 58),  
        ])

        
        delta = (nh_data[:, 1] * (nh_data[:, 2] - 1)).sum() / (nh_data[:, 2] - 1).sum()
        print("delta:", delta)
        for study_data, test_name in zip((nh_data, ci_data), ("calibrated", "calibrated_specres_1500_512_32_50")):
            x_hat = (study_data[:, 0] * study_data[:, 2]).sum() / study_data[:, 2].sum()
            s_hat = (study_data[:, 1] * (study_data[:, 2] - 1)).sum() / (study_data[:, 2] - 1).sum()
            snrs = result[result['test'] == test_name]['srt']
            print(test_name)
            print(x_hat, s_hat, study_data[:, 2].sum())
            print(np.mean(snrs), np.std(snrs), len(snrs), np.median(snrs))
            print(stats.ttest_ind_from_stats(
                mean1=x_hat, std1=s_hat, nobs1=study_data[:, 2].sum(), 
                mean2=np.mean(snrs), std2=np.std(snrs), nobs2=len(snrs),
                equal_var=True
            ))
            stat, p = stats.shapiro(snrs)
            print(f"Is my data normal: ({stat}, {p} > 0.05)")

            print(tost_from_summary(
                mean1=x_hat, std1=s_hat, nobs1=study_data[:, 2].sum(), 
                mean2=np.mean(snrs), std2=np.std(snrs), nobs2=len(snrs),
                delta=delta
            ))
            print()

    def pairwise(self, result):
        for x, y in itertools.combinations(set(result['test']), 2):
            g1 = result.loc[result['test'] == x, 'srt']
            g2 = result.loc[result['test'] == y, 'srt']
            mean1, std1, nobs1 = np.mean(g1), np.std(g1), len(g1)
            mean2, std2, nobs2 = np.mean(g2), np.std(g2), len(g2)

            t, p = stats.ttest_ind_from_stats(mean1, std1, nobs1,
                                            mean2, std2, nobs2,
                                            equal_var=False)
            print(x, y, t, p, abs(mean1 - mean2))



    def handle(self, *args, **options):
        tests = Test.objects.filter(active=True)
        result = []
        for test in tests:
            for responses in test.iter_entries(True):
                levels = np.array([x.stimulus.level for x in responses])
                snr_levels = np.r_[
                    levels[-20:], Response.get_next_level(responses.last())
                ]
                questionary = responses[0].questionary
                result.append((
                    test.audio_generator,   
                    questionary.pk,
                    questionary.age,
                    questionary.first_time,
                    np.mean(snr_levels)
                ))
        result = pd.DataFrame(result, columns=["test", "pk", "age", "first_time", "srt"])
        m = result.groupby(["pk"])['test'].count() != 3
        not_completed = m[m].index.values
        deaf_people = result[(result['srt'] > -6.0) & (result['test'] == 'calibrated')]['pk'].values
        drop_pk = np.unique(np.r_[deaf_people, not_completed])

        print(result[result['pk'].isin(drop_pk)])
        result = result[~result['pk'].isin(drop_pk)]
        result.to_csv(settings.MEDIA_ROOT / "results.csv", index=False)
        self.vs_studies(result)
        # self.pairwise(result)
        # self.make_boxplot(result)
        # self.make_age_scatter_plot(result)
        # self.make_comparison_plot(result) 
