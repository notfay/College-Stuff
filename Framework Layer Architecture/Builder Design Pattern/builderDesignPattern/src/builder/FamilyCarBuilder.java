package builder;

import model.car.FamilyCar;
import model.engine.Engine;
import model.wheel.Wheel;

public class FamilyCarBuilder implements CarBuilder{

	private FamilyCar familyCar;
	
	
	public FamilyCarBuilder() {
		familyCar = new FamilyCar();
	}

	
	
	@Override
	public void setEngine(Engine engine) {
		familyCar.setEngine(engine);
		
	}

	@Override
	public void setWheel(Wheel wheel) {
		familyCar.setWheel(wheel);
		
	}
	
	
	public FamilyCar getCar() {
		return familyCar;
	}
	
	
	
	
	
}
