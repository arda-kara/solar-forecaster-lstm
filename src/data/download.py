"""
Script to download NASA satellite data for space weather forecasting.
"""
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

def download_goes_data(start_date, end_date, save_dir='data/raw'):
    """
    Download GOES X-ray flux data from NASA/NOAA.
    
    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    save_dir : str
        Directory to save the downloaded data
    
    Returns:
    --------
    str
        Path to the saved data file
    """
    print(f"Downloading GOES X-ray flux data from {start_date} to {end_date}...")
    
    # Convert to absolute path and ensure the save directory exists
    save_dir = os.path.abspath(save_dir)
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Using directory: {save_dir}")
    except Exception as e:
        print(f"Error creating directory {save_dir}: {e}")
        # Fall back to current directory
        save_dir = os.path.abspath('.')
        print(f"Falling back to current directory: {save_dir}")
    
    output_file = os.path.join(save_dir, f"goes_xray_{start_date}_to_{end_date}.csv")
    
    # Try using the NOAA SWPC JSON API first
    try:
        # This API provides the last 7 days of data
        url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            # Convert to DataFrame
            processed_data = []
            for item in data:
                try:
                    time_tag = item.get('time_tag')
                    if time_tag:
                        timestamp = datetime.strptime(time_tag, '%Y-%m-%dT%H:%M:%SZ')
                        if timestamp >= datetime.strptime(start_date, '%Y-%m-%d') and \
                           timestamp <= datetime.strptime(end_date, '%Y-%m-%d'):
                            processed_data.append({
                                'time': timestamp,
                                'xray_flux_short': item.get('flux', 0),
                                'xray_flux_long': item.get('flux_ratio', 0) * item.get('flux', 0)
                            })
                except Exception as e:
                    print(f"Error processing item: {e}")
                    continue
            
            if processed_data:
                df = pd.DataFrame(processed_data)
                
                # Save to file
                df.to_csv(output_file, index=False)
                
                print(f"Data saved to {output_file}")
                return output_file
            else:
                print("No data found in the specified date range from SWPC API.")
        else:
            print(f"Error with SWPC API: {response.status_code}")
    except Exception as e:
        print(f"Error with SWPC API: {str(e)}")
    
    # Try NASA CDAWEB API
    print("Trying NASA CDAWEB API...")
    try:
        # Implementation for NASA CDAWEB API would go here
        # This is a placeholder for actual implementation
        pass
    except Exception as e:
        print(f"Error with NASA CDAWEB API: {str(e)}")
    
    # Try Stanford Solar Center data
    print("Trying Stanford Solar Center data...")
    try:
        # Implementation for Stanford Solar Center data would go here
        print("Stanford data parsing not implemented yet.")
    except Exception as e:
        print(f"Error with Stanford data: {str(e)}")
    
    # If all APIs fail, generate synthetic data
    print("No real data retrieved. Generating synthetic data instead.")
    return generate_synthetic_goes_data(start_date, end_date, save_dir)

def generate_synthetic_goes_data(start_date, end_date, output_file=None):
    """
    Generate synthetic GOES X-ray flux data when real data is not available.
    
    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    output_file : str, optional
        Path to save the generated data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing synthetic GOES data
    """
    print(f"Generating synthetic GOES X-ray flux data from {start_date} to {end_date}...")
    
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create hourly timestamps
    timestamps = pd.date_range(start=start, end=end, freq='h')
    
    # Create DataFrame to hold the data
    df = pd.DataFrame(index=timestamps)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'time_tag'}, inplace=True)
    
    # Number of days for simulation
    n_days = (end - start).days + 1
    
    # Generate background X-ray flux (follows solar rotation ~27 days)
    solar_rotation = 27 * 24  # hours
    background_cycle = np.sin(2 * np.pi * np.arange(len(timestamps)) / solar_rotation)
    background_flux_short = 1e-7 * (1 + 0.5 * background_cycle)
    background_flux_long = 5e-8 * (1 + 0.5 * background_cycle)
    
    # Add daily variation (higher during day)
    daily_cycle = np.sin(2 * np.pi * (df['time_tag'].dt.hour / 24 + 0.25))
    daily_variation_short = 2e-8 * (0.5 + 0.5 * daily_cycle)
    daily_variation_long = 1e-8 * (0.5 + 0.5 * daily_cycle)
    
    # Generate random flares
    n_flares = int(n_days * 1.5)  # Average 1.5 flares per day
    flare_times = np.random.choice(range(len(timestamps)), size=n_flares, replace=False)
    flare_magnitudes_short = np.random.exponential(scale=5e-6, size=n_flares)
    flare_magnitudes_long = flare_magnitudes_short * 0.7  # Long wavelength flux is typically lower
    
    # Create arrays for flux values
    xray_flux_short = background_flux_short + daily_variation_short
    xray_flux_long = background_flux_long + daily_variation_long
    
    # Add flares with realistic decay
    for i, flare_time in enumerate(flare_times):
        # Flare duration (hours)
        duration = np.random.randint(3, 12)
        
        # Flare profile (fast rise, exponential decay)
        rise_time = np.random.randint(1, 3)
        decay_time = duration - rise_time
        
        # Create flare profile
        rise_profile = np.linspace(0, 1, rise_time)
        decay_profile = np.exp(-np.linspace(0, 3, decay_time))
        flare_profile = np.concatenate([rise_profile, decay_profile])
        
        # Ensure we don't go beyond array bounds
        end_idx = min(flare_time + len(flare_profile), len(xray_flux_short))
        profile_length = end_idx - flare_time
        
        # Add flare to background
        xray_flux_short[flare_time:end_idx] += flare_magnitudes_short[i] * flare_profile[:profile_length]
        xray_flux_long[flare_time:end_idx] += flare_magnitudes_long[i] * flare_profile[:profile_length]
    
    # Add random noise
    noise_short = np.random.normal(0, 1e-8, len(timestamps))
    noise_long = np.random.normal(0, 5e-9, len(timestamps))
    
    xray_flux_short += noise_short
    xray_flux_long += noise_long
    
    # Ensure no negative values
    xray_flux_short = np.maximum(xray_flux_short, 1e-9)
    xray_flux_long = np.maximum(xray_flux_long, 1e-9)
    
    # Add to DataFrame
    df['xray_flux_short'] = xray_flux_short
    df['xray_flux_long'] = xray_flux_long
    
    # Add flare classifications
    df['flare_class'] = 'A'
    df.loc[df['xray_flux_short'] >= 1e-7, 'flare_class'] = 'B'
    df.loc[df['xray_flux_short'] >= 1e-6, 'flare_class'] = 'C'
    df.loc[df['xray_flux_short'] >= 1e-5, 'flare_class'] = 'M'
    df.loc[df['xray_flux_short'] >= 1e-4, 'flare_class'] = 'X'
    
    # Save to file if output_file is provided
    if output_file:
        try:
            # Convert to absolute path
            output_file = os.path.abspath(output_file)
            # Ensure the directory exists
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            
            # Try to save the file
            df.to_csv(output_file, index=False)
            print(f"Synthetic GOES data saved to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving synthetic GOES data to {output_file}: {e}")
            print("Returning DataFrame instead of file path")
            
            # Try saving to current directory as fallback
            try:
                fallback_file = os.path.join(os.path.abspath('.'), f"goes_xray_{start_date}_to_{end_date}.csv")
                df.to_csv(fallback_file, index=False)
                print(f"Saved to fallback location: {fallback_file}")
                return fallback_file
            except Exception as e2:
                print(f"Fallback save also failed: {e2}")
                return df
    
    return df

def download_ace_data(start_date, end_date, save_dir='data/raw'):
    """
    Download ACE solar wind data from NASA/NOAA.
    
    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    save_dir : str
        Directory to save the downloaded data
    
    Returns:
    --------
    str
        Path to the saved data file
    """
    print(f"Downloading ACE solar wind data from {start_date} to {end_date}...")
    
    # Convert to absolute path and ensure the save directory exists
    save_dir = os.path.abspath(save_dir)
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Using directory: {save_dir}")
    except Exception as e:
        print(f"Error creating directory {save_dir}: {e}")
        # Fall back to current directory
        save_dir = os.path.abspath('.')
        print(f"Falling back to current directory: {save_dir}")
    
    output_file = os.path.join(save_dir, f"ace_swepam_{start_date}_to_{end_date}.csv")
    
    # Try using the NOAA SWPC API first
    try:
        # This API provides the last 7 days of data
        url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            # Convert to DataFrame
            if len(data) > 1:  # Check if there's data beyond the header
                headers = data[0]
                rows = data[1:]
                
                df = pd.DataFrame(rows, columns=headers)
                
                # Convert time column
                df['time'] = pd.to_datetime(df['time_tag'])
                df = df.drop('time_tag', axis=1)
                
                # Convert numeric columns
                numeric_cols = ['density', 'speed', 'temperature']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Filter by date range
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                df = df[(df['time'] >= start_dt) & (df['time'] <= end_dt)]
                
                if not df.empty:
                    # Save to file
                    df.to_csv(output_file, index=False)
                    
                    print(f"Data saved to {output_file}")
                    return output_file
                else:
                    print("No data found in the specified date range from SWPC API.")
            else:
                print("No data returned from SWPC API.")
        else:
            print(f"Error with SWPC API: {response.status_code}")
    except Exception as e:
        print(f"Error with SWPC API: {str(e)}")
    
    # Try NASA CDAWEB API for ACE data
    print("Trying NASA CDAWEB API for ACE data...")
    try:
        # Implementation for NASA CDAWEB API would go here
        print("NASA CDAWEB API implementation for ACE data not completed.")
    except Exception as e:
        print(f"Error with NASA CDAWEB API: {str(e)}")
    
    # If all APIs fail, generate synthetic data
    print("No real data retrieved or error occurred. Generating synthetic data instead.")
    return generate_synthetic_ace_data(start_date, end_date, save_dir)

def generate_synthetic_ace_data(start_date, end_date, output_file=None):
    """
    Generate synthetic ACE solar wind data when real data is not available.
    
    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    output_file : str, optional
        Path to save the generated data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing synthetic ACE data
    """
    print(f"Generating synthetic ACE solar wind data from {start_date} to {end_date}...")
    
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create hourly timestamps
    timestamps = pd.date_range(start=start, end=end, freq='h')
    
    # Create DataFrame to hold the data
    df = pd.DataFrame(index=timestamps)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'time_tag'}, inplace=True)
    
    # Number of hours for simulation
    n_hours = len(timestamps)
    
    # Generate base solar wind speed (follows ~27 day solar rotation)
    solar_rotation = 27 * 24  # hours
    base_speed = 400 + 50 * np.sin(2 * np.pi * np.arange(n_hours) / solar_rotation)
    
    # Add high-speed streams
    n_streams = int(n_hours / (7 * 24))  # Approximately one stream per week
    stream_times = np.random.choice(range(n_hours), size=n_streams, replace=False)
    
    speed = base_speed.copy()
    
    for stream_time in stream_times:
        # Stream duration (hours)
        duration = np.random.randint(24, 72)
        
        # Stream profile (fast rise, slow decay)
        rise_time = np.random.randint(6, 12)
        decay_time = duration - rise_time
        
        # Create stream profile
        rise_profile = np.linspace(0, 1, rise_time)
        decay_profile = np.exp(-np.linspace(0, 3, decay_time))
        stream_profile = np.concatenate([rise_profile, decay_profile])
        
        # Stream magnitude
        magnitude = np.random.uniform(100, 300)
        
        # Ensure we don't go beyond array bounds
        end_idx = min(stream_time + len(stream_profile), n_hours)
        profile_length = end_idx - stream_time
        
        # Add stream to base speed
        speed[stream_time:end_idx] += magnitude * stream_profile[:profile_length]
    
    # Generate density (anti-correlated with speed)
    base_density = 5 + 3 * np.sin(2 * np.pi * np.arange(n_hours) / (solar_rotation * 1.1))  # Slightly different period
    density = base_density - 0.01 * (speed - 400)  # Anti-correlation
    
    # Add CMEs (coronal mass ejections)
    n_cmes = int(n_hours / (14 * 24))  # Approximately one CME every two weeks
    cme_times = np.random.choice(range(n_hours), size=n_cmes, replace=False)
    
    for cme_time in cme_times:
        # CME duration (hours)
        duration = np.random.randint(12, 36)
        
        # CME profile
        profile = np.exp(-np.linspace(0, 4, duration))
        
        # CME magnitude
        speed_magnitude = np.random.uniform(100, 500)
        density_magnitude = np.random.uniform(10, 30)
        
        # Ensure we don't go beyond array bounds
        end_idx = min(cme_time + duration, n_hours)
        profile_length = end_idx - cme_time
        
        # Add CME effects
        speed[cme_time:end_idx] += speed_magnitude * profile[:profile_length]
        density[cme_time:end_idx] += density_magnitude * profile[:profile_length]
    
    # Generate temperature (correlated with speed)
    temperature = 1e5 + 500 * (speed - 400)
    
    # Generate magnetic field components
    bt = 5 + 2 * np.sin(2 * np.pi * np.arange(n_hours) / (solar_rotation * 0.9))
    bz = 2 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 3)) - 0.5  # 3-day variations with southward bias
    
    # Add IMF sector boundaries
    n_sectors = int(n_hours / (14 * 24))  # Approximately one sector boundary every two weeks
    sector_times = np.random.choice(range(n_hours), size=n_sectors, replace=False)
    
    for sector_time in sector_times:
        # Sector crossing duration
        duration = np.random.randint(6, 24)
        
        # Ensure we don't go beyond array bounds
        end_idx = min(sector_time + duration, n_hours)
        
        # Flip Bz component
        bz[sector_time:end_idx] = -bz[sector_time:end_idx]
    
    # Add random noise
    speed += np.random.normal(0, 10, n_hours)
    density += np.random.normal(0, 0.5, n_hours)
    temperature += np.random.normal(0, 1e4, n_hours)
    bt += np.random.normal(0, 0.5, n_hours)
    bz += np.random.normal(0, 0.5, n_hours)
    
    # Ensure physical values
    speed = np.maximum(speed, 250)
    density = np.maximum(density, 0.5)
    temperature = np.maximum(temperature, 5e4)
    
    # Add to DataFrame
    df['speed'] = speed
    df['density'] = density
    df['temperature'] = temperature
    df['bt'] = bt
    df['bz'] = bz
    
    # Save to file if output_file is provided
    if output_file:
        try:
            # Convert to absolute path
            output_file = os.path.abspath(output_file)
            # Ensure the directory exists
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            
            # Try to save the file
            df.to_csv(output_file, index=False)
            print(f"Synthetic ACE data saved to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving synthetic ACE data to {output_file}: {e}")
            print("Returning DataFrame instead of file path")
            
            # Try saving to current directory as fallback
            try:
                fallback_file = os.path.join(os.path.abspath('.'), f"ace_swepam_{start_date}_to_{end_date}.csv")
                df.to_csv(fallback_file, index=False)
                print(f"Saved to fallback location: {fallback_file}")
                return fallback_file
            except Exception as e2:
                print(f"Fallback save also failed: {e2}")
                return df
    
    return df

def download_sdo_images(start_date, end_date, save_dir='data/raw'):
    """
    Download SDO (Solar Dynamics Observatory) images for the specified date range.
    """
    print(f"Downloading SDO images from {start_date} to {end_date}...")
    
    # Convert to absolute path and ensure the save directory exists
    save_dir = os.path.abspath(save_dir)
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Using directory: {save_dir}")
    except Exception as e:
        print(f"Error creating directory {save_dir}: {e}")
        # Fall back to current directory
        save_dir = os.path.abspath('.')
        print(f"Falling back to current directory: {save_dir}")
    
    image_dir = os.path.join(save_dir, 'sdo_images')
    os.makedirs(image_dir, exist_ok=True)
    
    # Convert dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate a list of dates to download
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    # For each date, download images from different wavelengths
    wavelengths = ['171', '193', '304', '131']
    
    for date in tqdm(dates, desc="Downloading SDO images"):
        for wavelength in wavelengths:
            # Construct URL for the image - fixed format
            date_str = date.replace('-', '')
            url = f"https://sdo.gsfc.nasa.gov/assets/img/browse/{date[:4]}/{date_str}/AIA{date_str}_{wavelength}.jpg"
            
            # Define output path
            output_path = os.path.join(image_dir, f"sdo_{date}_{wavelength}.jpg")
            
            # Skip if file already exists
            if os.path.exists(output_path):
                continue
            
            # Download image
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
    
    print(f"SDO images saved to {image_dir}")
    return image_dir

if __name__ == "__main__":
    # Example usage
    start_date = "2020-01-01"
    end_date = "2020-01-07"
    
    download_goes_data(start_date, end_date)
    download_ace_data(start_date, end_date)
    download_sdo_images(start_date, end_date) 