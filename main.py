"""
Mkdocs-macros module for FlySearch documentation.
Defines custom macros and filters for the documentation site.
"""


def define_env(env):
    """
    Define macros, filters, and variables for mkdocs-macros.
    
    This function is called by mkdocs-macros plugin when building the site.
    """
    
    @env.macro
    def render_leaderboard(df, groups, pretty_names):
        """
        Render the leaderboard table with computed overall scores.
        
        Args:
            df: Pandas DataFrame with leaderboard data
            groups: List of tuples (group_label, [metric_keys])
            pretty_names: Dictionary mapping metric keys to pretty display names
            
        Returns:
            HTML string containing the leaderboard table
        """
        records = df.to_dict(orient='records')
        
        # Compute column maxima while ignoring NaNs
        maxima = {}
        for group_label, metrics in groups:
            for m in metrics:
                if m in df.columns:
                    colmax = df[m].max()
                else:
                    colmax = None
                maxima[m] = colmax
        
        # Compute overall scores for each record - average of group averages
        overall_scores = {}
        for r in records:
            model = r.get("model")
            group_averages = []
            
            # Calculate average for each group
            for group_label, metrics in groups:
                group_values = []
                for metric in metrics:
                    val = r.get(metric)
                    if val is not None and val == val:  # Check for not None and not NaN
                        group_values.append(val)
                    else:
                        group_values.append(0.0) # missing results as zeros.
                
                # If group has any valid values, compute its average
                if len(group_values) > 0:
                    group_avg = sum(group_values) / len(group_values)
                    group_averages.append(group_avg)
                else:
                    group_averages.append(0.0) # missing results as zeros.
            
            # Compute overall as average of group averages
            if len(group_averages) > 0:
                overall_avg = sum(group_averages) / len(group_averages)
                overall_scores[model] = overall_avg
        
        # Find maximum overall score
        overall_max = None
        for model, score in overall_scores.items():
            if overall_max is None or score > overall_max:
                overall_max = score
        
        # Sort records by overall score (descending)
        sorted_records = sorted(records, 
                               key=lambda r: overall_scores.get(r.get("model"), -float('inf')), 
                               reverse=True)
        
        # Generate HTML table
        html = []
        html.append('<div style="text-align: center" markdown="1">')
        html.append('<div class="md-typeset__table" markdown="1">')
        html.append('<style>')
        html.append('th[colspan="2"] {')
        html.append('text-align: center !important;')
        html.append('}')
        html.append('td {')
        html.append('white-space: nowrap;')
        html.append('}')
        html.append('</style>')
        html.append('<table markdown="1">')
        html.append('  <thead>')
        html.append('    <tr>')
        html.append('      <th>Agent</th>')
        html.append('      <th>Progress</th>')
        
        # Group headers
        for group_label, metrics in groups:
            html.append(f'      <th colspan="{len(metrics)}">{group_label}</th>')
        html.append('    </tr>')
        html.append('    <tr>')
        html.append('      <th></th>')
        html.append('      <th data-sort-default>Overall (%)</th>')
        
        # Metric headers
        for group_label, metrics in groups:
            for m in metrics:
                pretty = pretty_names.get(m, m)
                html.append(f'      <th>{pretty}</th>')
        html.append('    </tr>')
        html.append('  </thead>')
        
        # Table body
        html.append('  <tbody>')
        
        # Human baseline row
        html.append('    <tr data-sort-method="none">')
        html.append('    <td>🧍 Human (untrained)</td>')
        html.append('    <td>—</td>')
        html.append('    <td>—</td>')
        html.append('    <td>66.7&nbsp;±&nbsp;4.5</td>')
        html.append('    <td>—</td>')
        html.append('    <td>—</td>')
        html.append('    <td>60.8&nbsp;±&nbsp;6.9</td>')
        html.append('    </tr>')
        
        # Model rows
        medal_icons = {
            0: '🥇 ',  # Gold medal for 1st place
            1: '🥈 ',  # Silver medal for 2nd place
            2: '🥉 '   # Bronze medal for 3rd place
        }
        
        for idx, r in enumerate(sorted_records):
            html.append('    <tr>')
            model = r.get("model")
            medal = medal_icons.get(idx, '')
            html.append(f'      <td>{medal}🤖 {model if model else "—"}</td>')
            
            # Overall column - computed dynamically
            overall_value = overall_scores.get(model)
            if overall_value is None:
                html.append('      <td>&mdash;</td>')
            else:
                overall_str = f'{overall_value:.1f}'
                if overall_max is not None and overall_value == overall_max:
                    html.append(f'      <td><strong>{overall_str}</strong></td>')
                else:
                    html.append(f'      <td>{overall_str}</td>')
            
            # Metric columns
            for group_label, metrics in groups:
                for metric in metrics:
                    value = r.get(metric)
                    std = r.get(metric + "-std")
                    
                    # Detect missing values: NaN or None
                    value_missing = (value is None) or (value != value)
                    std_missing = (std is None) or (std != std)
                    
                    if value_missing:
                        html.append('      <td>&mdash;</td>')
                    else:
                        # Format numeric value
                        if isinstance(value, (int, float)):
                            vstr = f'{value:.1f}'
                        else:
                            vstr = str(value)
                        
                        # Format std if present
                        if not std_missing and isinstance(std, (int, float)):
                            sstr = f'{std:.1f}'
                        else:
                            sstr = None
                        
                        # Bold if value is maximum
                        maxv = maxima.get(metric)
                        if maxv is not None and value == maxv:
                            cell_content = f'<strong>{vstr}</strong>'
                        else:
                            cell_content = vstr
                        
                        if sstr is not None:
                            cell_content += f'&nbsp;±&nbsp;{sstr}'
                        
                        html.append(f'      <td>{cell_content}</td>')
            
            html.append('    </tr>')
        
        html.append('  </tbody>')
        html.append('</table>')
        html.append('</div>')
        html.append('</div>')
        
        return ''.join(html)

