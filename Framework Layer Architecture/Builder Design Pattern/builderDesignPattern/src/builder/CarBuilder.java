package builder;

import model.engine.Engine;
import model.wheel.Wheel;

public interface CarBuilder {
		
	public void setEngine(Engine engine);
	public void setWheel(Wheel wheel);
	
	
}
