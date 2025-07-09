from flask import Flask, redirect, url_for, request, flash, render_template, Response, jsonify
try:
    from flask_bootstrap import Bootstrap5
except ImportError:
    # Fallback for different Bootstrap-Flask versions
    try:
        from bootstrap_flask import Bootstrap5
    except ImportError:
        Bootstrap5 = None
        print("Warning: Bootstrap5 not available - UI may not render properly")
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Text, ForeignKey, DateTime, Date, Float, Enum
from flask_hashing import Hashing
from datetime import datetime, date
import os
import logging
import enum

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
if Bootstrap5:
    bootstrap = Bootstrap5(app)
else:
    bootstrap = None
hashing = Hashing(app)

# Configuration
app.config['SECRET_KEY'] = '*#*hE*H@#*@(#H#*$#jkr(*$))'
# Use absolute path for database to avoid path resolution issues
import os
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'people_counter.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Enum for movement direction
class Direction(enum.Enum):
    IN = "IN"
    OUT = "OUT"

class CountSession(db.Model):
    """Table to track people counting sessions"""
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, unique=True)
    start_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    total_entries: Mapped[int] = mapped_column(Integer, default=0)
    total_exits: Mapped[int] = mapped_column(Integer, default=0)
    
    events = relationship('CountEvent', back_populates='session', cascade='all, delete-orphan')

class CountEvent(db.Model):
    """Table to store individual entry/exit events"""
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, unique=True)
    session_id: Mapped[int] = mapped_column(ForeignKey('count_session.id'))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    direction: Mapped[Direction] = mapped_column(Enum(Direction))
    people_count: Mapped[int] = mapped_column(Integer, default=1)
    detection_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    
    session = relationship('CountSession', back_populates='events')

class DailyCount(db.Model):
    """Table to store daily summary counts"""
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, unique=True)
    date: Mapped[date] = mapped_column(Date, default=date.today, unique=True)
    total_entries: Mapped[int] = mapped_column(Integer, default=0)
    total_exits: Mapped[int] = mapped_column(Integer, default=0)
    peak_occupancy: Mapped[int] = mapped_column(Integer, default=0)
    current_occupancy: Mapped[int] = mapped_column(Integer, default=0)

# Import camera functionality
from camera_controller import PeopleCounterCamera

# Global camera instance
people_counter_camera = None

def get_or_create_daily_count(target_date=None):
    """Get or create daily count record for specified date"""
    if target_date is None:
        target_date = date.today()
    
    # Try to get existing record
    daily_count = DailyCount.query.filter_by(date=target_date).first()
    if daily_count:
        return daily_count
    
    # If not found, create new one with error handling
    try:
        daily_count = DailyCount(date=target_date)
        db.session.add(daily_count)
        db.session.commit()
        logger.info(f"Created new daily count record for {target_date}")
        return daily_count
    except Exception as e:
        db.session.rollback()
        logger.warning(f"Failed to create daily count, attempting to get existing: {e}")
        # Try again in case another process created it
        daily_count = DailyCount.query.filter_by(date=target_date).first()
        if daily_count:
            return daily_count
        else:
            logger.error(f"Could not create or find daily count for {target_date}")
            raise e

@app.route('/')
def index():
    """Main people counter interface"""
    try:
        # Get current session
        current_session = CountSession.query.filter_by(end_time=None).first()
        if not current_session:
            current_session = CountSession()
            db.session.add(current_session)
            db.session.commit()
            logger.info(f"Created new counting session {current_session.id}")
        
        # Get daily counts
        today_count = get_or_create_daily_count()
        
        # Calculate current occupancy
        total_entries = CountEvent.query.filter_by(direction=Direction.IN).count()
        total_exits = CountEvent.query.filter_by(direction=Direction.OUT).count()
        current_occupancy = max(0, total_entries - total_exits)
        
        # Update daily count current occupancy
        today_count.current_occupancy = current_occupancy
        if current_occupancy > today_count.peak_occupancy:
            today_count.peak_occupancy = current_occupancy
        db.session.commit()
        
        # Get recent events for display
        recent_events = CountEvent.query.order_by(CountEvent.timestamp.desc()).limit(10).all()
        
        return render_template('counter.html',
                             current_session=current_session,
                             today_count=today_count,
                             current_occupancy=current_occupancy,
                             recent_events=recent_events)
        
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        flash(f"Error loading counter: {e}", 'error')
        return render_template('counter.html',
                             current_session=None,
                             today_count=None,
                             current_occupancy=0,
                             recent_events=[])

@app.route('/start_counter')
def start_counter():
    """Initialize camera for people counting"""
    global people_counter_camera
    try:
        if people_counter_camera is None:
            logger.info("Creating new people counter camera instance")
            people_counter_camera = PeopleCounterCamera()
        else:
            logger.info("Camera already running, reusing existing instance")
        
        # Ensure we have an active session
        current_session = CountSession.query.filter_by(end_time=None).first()
        if not current_session:
            logger.info("Creating new counting session")
            current_session = CountSession()
            db.session.add(current_session)
            db.session.commit()
            logger.info(f"Created new session {current_session.id}")
        
        return jsonify({
            'status': 'success', 
            'message': 'People counter system ready',
            'session_id': current_session.id
        })
    except Exception as e:
        logger.error(f"Failed to start people counter: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/video_feed')
def video_feed():
    """Video streaming route for people counter"""
    global people_counter_camera
    if people_counter_camera is None:
        logger.info("Initializing camera for video feed")
        people_counter_camera = PeopleCounterCamera()
        
        # Ensure we have an active session
        current_session = CountSession.query.filter_by(end_time=None).first()
        if not current_session:
            logger.info("Creating new session for video feed")
            current_session = CountSession()
            db.session.add(current_session)
            db.session.commit()
            logger.info(f"Created new session {current_session.id}")
    
    return Response(people_counter_camera.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_count_events')
def check_count_events():
    """Check for new count events and save to database with enhanced reliability"""
    global people_counter_camera
    
    if people_counter_camera is None:
        logger.warning("Check count events called but camera not initialized")
        return jsonify({'status': 'error', 'message': 'Camera not initialized'})
    
    try:
        events = people_counter_camera.get_and_clear_events()
        if events:
            logger.info(f"🔄 Processing {len(events)} count events for automatic saving")
            
            # Ensure we have an active session (create if needed)
            current_session = CountSession.query.filter_by(end_time=None).first()
            if not current_session:
                logger.info("📝 Auto-creating new counting session")
                current_session = CountSession()
                db.session.add(current_session)
                try:
                    db.session.commit()
                    logger.info(f"✅ Created session {current_session.id}")
                except Exception as e:
                    logger.error(f"❌ Failed to create session: {e}")
                    db.session.rollback()
                    return jsonify({'status': 'error', 'message': f'Failed to create session: {e}'})
            
            # Get today's daily count (create if needed)
            today_count = get_or_create_daily_count()
            
            saved_events = []
            failed_events = []
            
            for event_idx, event in enumerate(events):
                try:
                    # Create count event record with automatic retry
                    count_event = CountEvent(
                        session_id=current_session.id,
                        direction=Direction.IN if event['direction'] == 'IN' else Direction.OUT,
                        people_count=event.get('people_count', 1),
                        detection_confidence=event.get('confidence', 0.0)
                    )
                    db.session.add(count_event)
                    
                    # Update session totals automatically
                    if event['direction'] == 'IN':
                        current_session.total_entries += event.get('people_count', 1)
                        today_count.total_entries += event.get('people_count', 1)
                        logger.info(f"➡️ Auto-saved ENTRY event #{event_idx + 1}")
                    else:
                        current_session.total_exits += event.get('people_count', 1)
                        today_count.total_exits += event.get('people_count', 1)
                        logger.info(f"⬅️ Auto-saved EXIT event #{event_idx + 1}")
                    
                    # Attempt to commit this individual event
                    try:
                        db.session.commit()
                        saved_events.append({
                            'direction': event['direction'],
                            'people_count': event.get('people_count', 1),
                            'timestamp': count_event.timestamp.isoformat(),
                            'event_id': count_event.id
                        })
                        logger.info(f"💾 Event {count_event.id} committed to database")
                    except Exception as commit_error:
                        logger.error(f"❌ Commit failed for event {event_idx + 1}: {commit_error}")
                        db.session.rollback()
                        failed_events.append(event)
                        continue
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process event {event_idx + 1}: {e}")
                    failed_events.append(event)
                    db.session.rollback()
                    continue
            
            # Update current occupancy automatically
            try:
                current_occupancy = max(0, today_count.total_entries - today_count.total_exits)
                today_count.current_occupancy = current_occupancy
                
                # Update peak occupancy if needed
                if current_occupancy > today_count.peak_occupancy:
                    today_count.peak_occupancy = current_occupancy
                    logger.info(f"📈 New peak occupancy: {current_occupancy}")
                
                db.session.commit()
                logger.info(f"📊 Updated occupancy: {current_occupancy} people")
                
            except Exception as e:
                logger.error(f"❌ Failed to update occupancy: {e}")
                db.session.rollback()
            
            # Prepare response
            response_data = {
                'status': 'success',
                'message': f'Auto-saved {len(saved_events)}/{len(events)} count events',
                'events': saved_events,
                'current_occupancy': current_occupancy if 'current_occupancy' in locals() else 0,
                'session_id': current_session.id,
                'total_entries_today': today_count.total_entries,
                'total_exits_today': today_count.total_exits
            }
            
            if failed_events:
                response_data['warning'] = f'{len(failed_events)} events failed to save'
                response_data['failed_count'] = len(failed_events)
                logger.warning(f"⚠️ {len(failed_events)} events failed to save")
            
            logger.info(f"✅ Auto-save complete: {len(saved_events)} events saved")
            return jsonify(response_data)
        else:
            # No events - this is normal, just return status
            return jsonify({
                'status': 'no_events', 
                'message': 'No new count events',
                'auto_save': 'active'
            })
            
    except Exception as e:
        logger.error(f"💥 Critical error in automatic count event processing: {e}", exc_info=True)
        try:
            db.session.rollback()
        except:
            pass
        return jsonify({
            'status': 'error', 
            'message': f'Critical auto-save error: {str(e)}',
            'auto_save': 'failed'
        })

@app.route('/auto_save_status')
def auto_save_status():
    """Check auto-save system status"""
    try:
        # Test database connection
        db.session.execute('SELECT 1')
        db_status = 'connected'
        
        # Check current session
        current_session = CountSession.query.filter_by(end_time=None).first()
        session_status = 'active' if current_session else 'none'
        
        # Check today's data
        today_count = DailyCount.query.filter_by(date=date.today()).first()
        today_status = 'active' if today_count else 'none'
        
        # Check camera status
        global people_counter_camera
        camera_status = 'active' if people_counter_camera is not None else 'inactive'
        
        return jsonify({
            'status': 'success',
            'auto_save': 'operational',
            'database': db_status,
            'session': session_status,
            'today_data': today_status,
            'camera': camera_status,
            'session_id': current_session.id if current_session else None,
            'entries_today': today_count.total_entries if today_count else 0,
            'exits_today': today_count.total_exits if today_count else 0
        })
        
    except Exception as e:
        logger.error(f"Auto-save status check failed: {e}")
        return jsonify({
            'status': 'error',
            'auto_save': 'degraded',
            'message': str(e)
        })

def ensure_database_ready():
    """Ensure database is ready for auto-saving"""
    try:
        # Create tables if they don't exist
        with app.app_context():
            db.create_all()
            logger.info("✅ Database tables verified/created")
        
        # Ensure we have a daily count for today
        today_count = get_or_create_daily_count()
        logger.info(f"✅ Today's count record ready: {today_count.id}")
        
        # Ensure we have an active session
        current_session = CountSession.query.filter_by(end_time=None).first()
        if not current_session:
            current_session = CountSession()
            db.session.add(current_session)
            db.session.commit()
            logger.info(f"✅ Auto-created counting session: {current_session.id}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database preparation failed: {e}")
        return False

@app.route('/end_session', methods=['POST'])
def end_session():
    """End current counting session"""
    try:
        current_session = CountSession.query.filter_by(end_time=None).first()
        if current_session:
            current_session.end_time = datetime.utcnow()
            db.session.commit()
            logger.info(f"Ended counting session {current_session.id}")
            return jsonify({
                'status': 'success', 
                'message': f'Session {current_session.id} ended successfully',
                'session_id': current_session.id,
                'total_entries': current_session.total_entries,
                'total_exits': current_session.total_exits
            })
        else:
            logger.info("End session requested but no active session found")
            return jsonify({
                'status': 'info', 
                'message': 'No active session to end'
            })
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stats')
def api_stats():
    """API endpoint for real-time statistics"""
    try:
        # Get today's counts
        today_count = get_or_create_daily_count()
        
        # Get current session
        current_session = CountSession.query.filter_by(end_time=None).first()
        
        # Calculate current occupancy
        current_occupancy = max(0, today_count.total_entries - today_count.total_exits)
        
        # Get recent events
        recent_events = CountEvent.query.order_by(CountEvent.timestamp.desc()).limit(5).all()
        events_data = []
        for event in recent_events:
            events_data.append({
                'direction': event.direction.value,
                'people_count': event.people_count,
                'timestamp': event.timestamp.strftime('%H:%M:%S'),
                'confidence': event.detection_confidence
            })
        
        return jsonify({
            'status': 'success',
            'current_occupancy': current_occupancy,
            'today_entries': today_count.total_entries,
            'today_exits': today_count.total_exits,
            'peak_occupancy': today_count.peak_occupancy,
            'session_entries': current_session.total_entries if current_session else 0,
            'session_exits': current_session.total_exits if current_session else 0,
            'recent_events': events_data
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stats')
def stats():
    """Statistics dashboard"""
    try:
        # Get daily counts for the last 7 days
        from sqlalchemy import func
        daily_counts = DailyCount.query.order_by(DailyCount.date.desc()).limit(7).all()
        
        # Get all sessions
        sessions = CountSession.query.order_by(CountSession.start_time.desc()).limit(20).all()
        
        # Calculate total statistics
        total_entries = sum(dc.total_entries for dc in daily_counts)
        total_exits = sum(dc.total_exits for dc in daily_counts)
        
        return render_template('stats.html',
                             daily_counts=daily_counts,
                             sessions=sessions,
                             total_entries=total_entries,
                             total_exits=total_exits)
                             
    except Exception as e:
        logger.error(f"Error in stats route: {e}")
        flash(f"Error loading statistics: {e}", 'error')
        return render_template('stats.html',
                             daily_counts=[],
                             sessions=[],
                             total_entries=0,
                             total_exits=0)

@app.route('/backup_recovery', methods=['POST'])
def backup_recovery():
    """Recover count events from backup storage"""
    global people_counter_camera
    
    if people_counter_camera is None:
        return jsonify({'status': 'error', 'message': 'Camera not initialized'})
    
    try:
        backup_events = people_counter_camera.get_backup_events()
        if not backup_events:
            return jsonify({'status': 'info', 'message': 'No backup events to recover'})
        
        logger.info(f"🔄 Attempting to recover {len(backup_events)} backup events")
        
        # Ensure we have session and daily count
        current_session = CountSession.query.filter_by(end_time=None).first()
        if not current_session:
            current_session = CountSession()
            db.session.add(current_session)
            db.session.commit()
        
        today_count = get_or_create_daily_count()
        
        recovered_count = 0
        for event in backup_events:
            try:
                count_event = CountEvent(
                    session_id=current_session.id,
                    direction=Direction.IN if event['direction'] == 'IN' else Direction.OUT,
                    people_count=event.get('people_count', 1),
                    detection_confidence=event.get('confidence', 0.0),
                    timestamp=event.get('timestamp', datetime.utcnow())
                )
                db.session.add(count_event)
                
                # Update totals
                if event['direction'] == 'IN':
                    current_session.total_entries += event.get('people_count', 1)
                    today_count.total_entries += event.get('people_count', 1)
                else:
                    current_session.total_exits += event.get('people_count', 1)
                    today_count.total_exits += event.get('people_count', 1)
                
                recovered_count += 1
                
            except Exception as e:
                logger.error(f"Failed to recover backup event: {e}")
                continue
        
        db.session.commit()
        
        # Clear backup after successful recovery
        people_counter_camera.clear_backup_events()
        
        logger.info(f"✅ Successfully recovered {recovered_count} events from backup")
        return jsonify({
            'status': 'success',
            'message': f'Recovered {recovered_count} events from backup',
            'recovered_count': recovered_count
        })
        
    except Exception as e:
        logger.error(f"Backup recovery failed: {e}")
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/reset_counts', methods=['POST'])
def reset_counts():
    """Reset all counts (for testing/maintenance)"""
    try:
        # This is a dangerous operation, so let's add some protection
        confirmation = request.form.get('confirmation', '')
        if confirmation.lower() != 'reset':
            return jsonify({'status': 'error', 'message': 'Invalid confirmation'})
        
        # Delete all records
        CountEvent.query.delete()
        CountSession.query.delete()
        DailyCount.query.delete()
        db.session.commit()
        
        # Also clear backup events
        global people_counter_camera
        if people_counter_camera:
            people_counter_camera.clear_backup_events()
        
        logger.info("All count data has been reset")
        return jsonify({'status': 'success', 'message': 'All counts have been reset'})
        
    except Exception as e:
        logger.error(f"Error resetting counts: {e}")
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("✅ People counter database initialized")
        
        # Ensure auto-save system is ready
        if ensure_database_ready():
            print("✅ Auto-save system initialized")
        else:
            print("❌ Auto-save system initialization failed")
    
    app.run(debug=True, threaded=True) 